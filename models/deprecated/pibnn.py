import torch
import torch.nn as nn
import torch.distributions as dist
import torchbnn as bnn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
from copy import deepcopy
import os
import datetime

class PIBNN(nn.Module):
    def __init__(self, input_size=3, output_size=3, hidden_layer_size=128, prior_sigma_lay=0.01, sensor_measurement_uncertainty=1, lambda_params=None, epochs=1000, bias_in=True):
        super(PIBNN, self).__init__()
        self.epochs = epochs
        self.loss_values = []

        if type(lambda_params) is dict:
            self.lambda_ml = lambda_params['ml']
            self.lambda_kinematic_physics = lambda_params['physics']
        else:
            self.lambda_ml = 0.7
            self.lambda_kinematic_physics = 0.3

        self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma_lay, in_features=input_size, out_features=hidden_layer_size, bias=bias_in)
        self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma_lay, in_features=hidden_layer_size, out_features=hidden_layer_size, bias=bias_in)
        self.fc3 = bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma_lay, in_features=hidden_layer_size, out_features=output_size, bias=bias_in)

        self.activation = nn.GELU()
        self.criterion_1 = nn.MSELoss()  # Using Mean Squared Error (MSE) loss
        self.criterion_2 = bnn.BKLLoss(reduction='mean', last_layer_only=True)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)  # Adam optimizer

        state_size = 3
        self.physics_process_noise = torch.full((state_size,), sensor_measurement_uncertainty/1000)

        ## EKF OBSERVATION MODEL ##
        self.A_k_minus_1 = np.identity(state_size)
        self.Q_k = sensor_measurement_uncertainty/1000 * np.identity(state_size)  # State model process noise
        self.R_k = (sensor_measurement_uncertainty ** 2) * np.identity(state_size)  # Sensor uncertainty [sigma^2]

        self.H_k = np.identity(state_size)  # measurement matrix
        self.sensor_noise_w_k = np.full((state_size,), 0)  # no sensor bias
        self.sensor_measurement_uncertainty = sensor_measurement_uncertainty

    def forward_physics(self, x_obs, v_obs, dt_vec):
        return x_obs.add_(v_obs.mul_(dt_vec)).add_(self.physics_process_noise)

    def forward_ml(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

    def __physics_loss_simple_kinematics(self, position_pred, position_input, vel_true, dt):  # SIMPLE KINEMATICS BASED
        vel_pred = (position_pred - position_input)/dt
        displacement_error = dt*(vel_true-vel_pred)

        loss_physics = ((displacement_error[:, 0]) ** 2 +
                        (displacement_error[:, 1]) ** 2 +
                        (displacement_error[:, 2]) ** 2).mean()
        return loss_physics

    def __ekf_observation_model(self, state_estimate_k, z_k_observation_vector, p_k_minus_1):
        """
        Extended Kalman Filter observation model. Fuses noisy sensor measurement to
        create an optimal estimate of the state.
        """

        state_est_np = state_estimate_k.numpy()
        z_obs_np = z_k_observation_vector.numpy()

        P_k = self.A_k_minus_1 @ p_k_minus_1 @ self.A_k_minus_1.T + self.Q_k
        S_k = self.H_k @ P_k @ self.H_k.T + self.R_k
        K_k = P_k @ self.H_k.T @ np.linalg.pinv(S_k)

        measurement_residual_y_k = z_obs_np - ((self.H_k @ state_est_np) + self.sensor_noise_w_k)
        observation_model_state_estimate_k = state_est_np + (K_k @ measurement_residual_y_k)
        P_k = P_k - (K_k @ self.H_k @ P_k)
        return observation_model_state_estimate_k, P_k

    def fit(self, x_obs, v_obs, dt_vec, vel_true, X_feature_space, y_true):
        x_state_pred = self.forward_physics(x_obs, v_obs, dt_vec)
        full_X = result = torch.cat((x_state_pred, X_feature_space), dim=1)

        for epoch in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            y_pred = self.forward_ml(full_X)
            ml_loss = self.criterion_1(y_pred, y_true) + self.criterion_2(self) # MSE + KL Divergence
            physics_loss = self.__physics_loss_simple_kinematics(y_pred, x_obs, vel_true, dt_vec)
            loss = self.lambda_ml*ml_loss + self.lambda_kinematic_physics*physics_loss
            loss.backward()

            self.loss_values.append(loss.item())
            self.optimizer.step()

            sys.stderr.write(f"\r Epoch: {epoch + 1:4d}/{self.epochs:4d} | Loss: {loss.item():6.2f} ")
            sys.stderr.flush()
        return None

    def predict(self, x_obs, v_obs, dt_vec, X_features, z_obs, num_samples=100, apply_ekf=False):
        self.eval()
        predictions = []
        with torch.no_grad():
            x_state_pred = self.forward_physics(x_obs, v_obs, dt_vec)
            full_X = torch.cat((x_state_pred, X_features))
            for _ in range(num_samples):
                predictions.append(self.forward_ml(full_X).unsqueeze(0))

        batch_preds = torch.cat(predictions, dim=0)
        state_est_mu = batch_preds.mean(0)
        state_est_p = batch_preds.std(0)

        ## ADDING EKF UPDATE: returning in NUMPY
        if apply_ekf:
            predicted_state, predicted_cov = self.__ekf_observation_model(state_est_mu, z_obs, state_est_p.numpy())
        else:
            predicted_state = state_est_mu.numpy()
            predicted_cov = state_est_p.numpy()

        return predicted_state, predicted_cov

    def plot(self):
        if len(self.loss_values) > 0:
            epochs = range(1, self.epochs + 1)
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, self.loss_values, label='Training Loss', marker='o', linestyle='-', color='b', linewidth=2)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Epochs vs Loss')
            plt.legend()
            plt.grid(True)
            plt.show()

    def save_model_weights(self, file_prefix="TEST", directory='saved_models/pibnn'):
        if not os.path.exists(directory):
            os.makedirs(directory)

        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{file_prefix}_pibnn_model_{current_time}.pth"
        filepath = os.path.join(directory, filename)

        torch.save(self.state_dict(), filepath)
        print(f"Model weights saved to {filepath}")

        return filepath

    def load_model_weights(self, filepath):
        """Loads the model weights from a specified file."""
        if os.path.exists(filepath):
            self.load_state_dict(torch.load(filepath))
            self.eval()
            print(f"Model weights loaded from {filepath}")
        else:
            raise FileNotFoundError(f"No such file: {filepath}")
