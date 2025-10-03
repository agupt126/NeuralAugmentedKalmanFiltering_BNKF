import matplotlib.pyplot as plt
import sys
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torchbnn as bnn

class BNN(nn.Module):
    def __init__(self, input_size, output_size=3, prior_sigma_lay=0.01, hidden_layer_size=128, epochs=1000, show_info=False, bias_in=True):
        super(BNN, self).__init__()
        self.epochs = epochs
        self.loss_values = []
        hidden_layer_size = 128
        self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma_lay, in_features=input_size, out_features=hidden_layer_size, bias=bias_in)
        self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma_lay, in_features=hidden_layer_size, out_features=hidden_layer_size, bias=bias_in)
        self.fc3 = bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma_lay, in_features=hidden_layer_size, out_features=output_size, bias=bias_in)

        self.activation = nn.GELU()
        self.criterion_1 = nn.MSELoss()
        self.criterion_2 = bnn.BKLLoss(reduction='mean', last_layer_only=True)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

    def fit(self, X_mat, y_true):
        for epoch in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()

            y_pred = self.forward(X_mat)
            loss = self.criterion_1(y_pred, y_true) + self.criterion_2(self)
            loss.backward()
            self.optimizer.step()
            self.loss_values.append(loss.item())
            sys.stderr.write(f"\r Epoch: {epoch + 1:4d}/{self.epochs:4d} | Loss: {loss.item():6.2f} ")
            sys.stderr.flush()

        return loss.item()

    def predict(self, x, num_samples=100):
        self.eval()
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                predictions.append(self.forward(x).unsqueeze(0))
        return torch.cat(predictions, dim=0).mean(0), torch.cat(predictions, dim=0).std(0)

    def plot(self):
        if len(self.loss_values) > 0:
            epochs = range(1, self.epochs + 1)
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, self.loss_values, label='Training Loss', marker='o', linestyle='-', color='b', linewidth=1)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Epochs vs Loss')
            plt.legend()
            plt.grid(True)
            plt.show()

    def save_model_weights(self, file_prefix="TEST", directory='saved_weights/bnn'):
        if not os.path.exists(directory):
            os.makedirs(directory)

        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{file_prefix}_bnn_model_{current_time}.pth"
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