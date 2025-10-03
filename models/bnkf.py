import torch
import torch.nn as nn
import torchbnn as bnn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import datetime


class BNKF(nn.Module):
    def __init__(self, input_size=3, output_size=3, hidden_layer_size=64, sensor_noise_level='low',
                 prior_sigma_lay=0.01, lambda_params=None, epochs=1000, bias_in=True):

        super(BNKF, self).__init__()
        self.epochs = epochs
        self.loss_values = []

        if type(lambda_params) is dict:
            self.lambda_ml = lambda_params['ml']
            self.lambda_kinematic_physics = lambda_params['physics']
        else:
            self.lambda_ml = 0.7
            self.lambda_kinematic_physics = 0.3

        self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma_lay, in_features=input_size,
                                   out_features=hidden_layer_size, bias=bias_in)
        self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma_lay, in_features=hidden_layer_size,
                                   out_features=hidden_layer_size, bias=bias_in)
        self.fc3 = bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma_lay, in_features=hidden_layer_size,
                                   out_features=hidden_layer_size, bias=bias_in)
        self.fc4 = bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma_lay, in_features=hidden_layer_size,
                                   out_features=hidden_layer_size, bias=bias_in)
        self.fc5 = bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma_lay, in_features=hidden_layer_size,
                                   out_features=output_size, bias=bias_in)

        self.activation = nn.GELU()
        self.criterion_1 = nn.MSELoss()  # Using Mean Squared Error (MSE) loss
        self.criterion_2 = bnn.BKLLoss(reduction='mean', last_layer_only=True)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)  # Adam optimizer

        match sensor_noise_level.lower():
            case "low":
                self.sensor_measurement_uncertainty = 1
            case "mid":
                self.sensor_measurement_uncertainty = 10
            case "high":
                self.sensor_measurement_uncertainty = 100

        self.state_size = output_size
        self.H_k_model = np.identity(self.state_size)  # measurement matrix [assuming already converted]

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        return x

    def forward_physics(self, _pos, _vel, _dt):
        return torch.cat((_pos + (_vel * _dt), _vel), dim=1)

    def __physics_loss_simple_kinematics(self, position_pred, position_input, velocity_input,
                                         dt_input):  # SIMPLE KINEMATICS BASED
        physics_pred = position_input + (velocity_input * dt_input)
        displacement_error = position_pred - physics_pred
        loss_physics = torch.mean(torch.sum(displacement_error ** 2, dim=1))
        return loss_physics

    def __kalman_update(self, state_estimate_minus_1, p_k_minus_1, z_k_observation_vector, sensor_sigma_vec):
        """
        Extended Kalman Filter observation model. Fuses noisy sensor measurement to
        create an optimal estimate of the state.
        """
        state_est_minus_1_np = state_estimate_minus_1.numpy()
        p_k_minus_1_np = p_k_minus_1.numpy()
        z_obs_np = z_k_observation_vector.numpy()
        sensor_sigma_vec_np = sensor_sigma_vec.numpy()

        R_k = (sensor_sigma_vec_np ** 2)[:, None] * np.identity(6)
        P_k = p_k_minus_1_np
        H_k = np.tile(self.H_k_model, (P_k.shape[0], 1, 1))  # Expands H_k to shape (batch_size, 6, 6)

        S_k = H_k @ P_k @ np.transpose(H_k, (0, 2, 1)) + R_k
        K_k = P_k @ np.transpose(H_k, (0, 2, 1)) @ np.linalg.pinv(S_k)

        measurement_residual_y_k = z_obs_np - np.einsum('bij,bj->bi', H_k, state_est_minus_1_np)
        observation_model_state_estimate_k = state_est_minus_1_np + np.einsum('ijk,ik->ij', K_k, measurement_residual_y_k)
        P_k = P_k - (K_k @ H_k @ P_k)
        return observation_model_state_estimate_k, P_k

    def fit(self, X_feature_space, y_true, physics_metadata):
        pos_in, vel_in, dt_in = physics_metadata['inputPos'], physics_metadata['inputVel'], physics_metadata['dtVec'],
        for epoch in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            y_pred = self.forward(X_feature_space)
            loss = self.criterion_1(y_pred, y_true) + self.criterion_2(self)  # MSE + KL Divergence
            loss.backward()

            self.loss_values.append(loss.item())
            self.optimizer.step()

            sys.stderr.write(f"\r Epoch: {epoch + 1:4d}/{self.epochs:4d} | Loss: {loss.item():6.2f} ")
            sys.stderr.flush()
        return None

    def predict(self, X_features, physics_metadata, num_samples=100, apply_filter_update=False):
        self.eval()
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                predictions.append(self.forward(X_features).unsqueeze(0))

        batch_preds = torch.cat(predictions, dim=0)
        state_est_mu = batch_preds.mean(dim=0)
        x_centered = batch_preds - state_est_mu
        state_est_p = torch.einsum('nvc,nvd->vcd', x_centered, x_centered) / (batch_preds.shape[0] - 1)

        ## ADDING EKF UPDATE: returning in NUMPY
        if apply_filter_update:
            z_obs, sig_obs = physics_metadata['inputObs'], physics_metadata['inputRngErr']
            pos_in, vel_in, dt_in = physics_metadata['inputPos'], physics_metadata['inputVel'], physics_metadata['dtVec']
            state_est_fused = state_est_mu
            #state_est_physics = self.forward_physics(pos_in, vel_in, dt_in)
            #state_est_fused = self.lambda_ml*state_est_mu + self.lambda_kinematic_physics*state_est_physics
            predicted_state, predicted_cov = self.__kalman_update(state_est_fused, state_est_p, z_obs, sig_obs)
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
        return None
