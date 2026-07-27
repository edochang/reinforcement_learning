import numpy as np
import torch
import torch.nn as nn
from algo import ValueFunctionWithApproximation

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        # ✅TODO: implement this method
        class NN (nn.Module):
            def __init__(self, state_dims):
                super().__init__()
                # 4 Layer neural network:
                # - 1 input layer, 2 hidden layers, and 1 output layer with 3 connection between them
                # - The input layer has 32 neurons, the hidden layer has another 32 neurons
                # - 4 layers with 3 connections between them
                self.hidden = nn.Sequential(
                    nn.Linear(state_dims, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                )
                # Output layer with 1 neuron for value estimation
                self.output = nn.Linear(32, 1)  
    
            def forward(self, x):
                return self.output(self.hidden(x))

        self.model = NN(state_dims)

        # Use AdamOptimizer with beta1=0.9, beta2=0.999. For the learning rate alpha, you can assume the alpha is always 0.001.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))

    def __call__(self,s):
        # ✅TODO: implement this method
        # Set the model to evaluation mode
        self.model.eval()
        # Convert the state to a tensor and add a batch dimension
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        # Pass the state through NN model and return the value estimate as a float
        return self.model(s_tensor).item()

    def update(self,alpha,G,s_tau):
        # ✅TODO: implement this method
        # convert G (scalar) to a tensor and add a batch dimension to address shape warning
        G_tensor = torch.tensor(G, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Set the model to training mode
        self.model.train()
        self.optimizer.zero_grad()
        s_tau_tensor = torch.tensor(s_tau, dtype=torch.float32).unsqueeze(0)
        output = self.model(s_tau_tensor)
        loss = torch.nn.functional.mse_loss(output, G_tensor)
        loss.backward()
        self.optimizer.step()

        return None

