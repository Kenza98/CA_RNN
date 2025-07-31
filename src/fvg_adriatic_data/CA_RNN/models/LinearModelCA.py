import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

load_file = "synthetic_data_interpolation/data/ca02_lr.pt"
data = torch.load(load_file)
print(type(data))
X_train, Y_train = data["X_TRAIN"], data["Y_TRAIN"]
print(X_train.shape), print(Y_train.shape)

train_dataset = TensorDataset(X_train, Y_train)

nb_features = 1
learning_rate = 1e-3
num_epochs = 50
batch_size = 32
output_dim = 1
input_dim = 9

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

model = nn.Sequential(nn.Linear(input_dim, output_dim))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.train()

k = 0
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        # FOR SEQUENTIAL shape (batch_size, -1) = (batch_size, 9)
        x_batch = x_batch.view(
            x_batch.size(0), -1
        )  # \\TODO this operation should be done in prepare.py
        if k == 0:
            print(f"shape of input : {x_batch.shape}\n")
            print(f"shape of input : {y_batch.shape}\n")
            k = 1
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

data["model_state_dict"] = model.state_dict()
data["model_type"] = "Sequential_Linear"
# Save everything back to the same .pt file
torch.save(data, load_file)
print(f"Model saved to {load_file}")
