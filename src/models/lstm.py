import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)

        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]   # shape: (batch_size, hidden_dim)

        out = self.fc(last_output)         # shape: (batch_size, output_dim)
        return out