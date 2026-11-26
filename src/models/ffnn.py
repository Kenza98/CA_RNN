import torch.nn as nn
"""
Some implementation choices:
#1 tanh, not ReLU: same activation as the vanilla RNN, so the FFNN differs from the RNNs by recurrence only. Inputs are z-scored and the net is shallow, so saturation is not a concern.

#2 Xavier/Glorot init (standard pairing for tanh). it keeps activation variance roughly constant across layers, so tanh units start out of saturation.
However, NN in this experiment are shallow so gradient vanish is not a real problem.
"""

class FeedForwardNN(nn.Module):
    """
    A classic MLP that takes sequence length tau as input for comparability with RNN models and
    compliance with rollout tests signature for models.
    """

    def __init__(self, input_dim, width, output_dim, num_layers=1, seq_length=6):
        super().__init__()
        self.tau = seq_length
        layers = [nn.Flatten()]
        in_features = self.tau * input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_features, width), nn.Tanh()]
            in_features = width
        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.net[-1].weight)  # output layer is linear: gain 1

    def forward(self, x):
        return self.net(x)  # Shape (batch, output_dim)


