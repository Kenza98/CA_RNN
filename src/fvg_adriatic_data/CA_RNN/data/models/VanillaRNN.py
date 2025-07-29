import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import torch.nn.functional as F


# my RNN subclasses torch.nn.Module
class VanillaRNN(nn.Module):
    # This method initializes the layers of the model:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # super() alls the parent class constructor to set up the nn.Module infrastructure :
        super().__init__()

        self.rnn = nn.RNN(input_dim, hidden_dim, 4, batch_first=True)
        # my module has an RNN "brain"
        self.fc = nn.Linear(hidden_dim, output_dim)
        # my NN module has a fully connected / dense layer
        # -->i need this in the forward method to process the hidden state output from the RNN

    """
    forward() function is necessary in any nn.Module subclass.
    It specifies how the input x moves through the layers of the model.
    """

    def forward(self, x):
        out, _ = self.rnn(
            x
        )  # out contains the hidden states for each time step in the sequence
        # _ cuz i didn't need the final hidden state. I use the one from last time step at each run
        out = self.fc(out[:, -1, :])  # Use the last hidden state
        return out
