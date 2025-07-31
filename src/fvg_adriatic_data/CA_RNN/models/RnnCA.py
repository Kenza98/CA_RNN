import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, TensorDataset
import VanillaRNN as vrnn
import matplotlib.pyplot as plt


# "./data/cop_ml_ready.pt"
load_file = "data/cop_ml_ready.pt"
data = torch.load(load_file)

X, Y = data["X_TRAIN"], data["Y_TRAIN"]

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

# print(X.ndim, Y.ndim)  # 3, 2

nan_mask_X = torch.isnan(X).any(dim=(1, 2))  # Check for NaNs in each sample
nan_mask_Y = torch.isnan(Y).any(dim=(1))

nan_mask = nan_mask_X | nan_mask_Y

valid_indices = (~nan_mask).nonzero(as_tuple=True)[0]
# print(valid_indices.shape)


X_clean = X[valid_indices]
Y_clean = Y[valid_indices]

clean_dataset = TensorDataset(X_clean, Y_clean)

nb_features = 1
learning_rate = 1e-4
num_epochs = 10
batch_size = 32
output_dim = 1
input_dim = 9
seq_length = 4
hidden_dim = 7 * 8

train_loader = DataLoader(clean_dataset, batch_size, shuffle=True)
    
model = vrnn.VanillaRNN(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.train()

grad_history = {} #store gradient over epoch to plot smt, the log takes time...

for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        # FOR SEQUENTIAL shape (batch_size, -1) = (batch_size, 9)
        # print(f"shape of input : {x_batch.shape}\n")
        # print(f"shape of input : {y_batch.shape}\n")

        optimizer.zero_grad()
        y_pred = model(x_batch)
        # print(f"The target shape : {y_batch.shape}")
        # print(f"The output shape : {y_pred.shape}")

        loss = criterion(y_pred, y_batch)
        loss.backward() #propagate the gradients

        with torch.no_grad():
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if name not in grad_history:
                        grad_history[name] = []
                    grad_history[name].append(grad_norm)
            


        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

plt.figure(figsize=(10, 6))
for name, norms in grad_history.items():
    plt.plot(norms, label=name)

plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norms per Parameter Across Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gradient_norms.png")
plt.show()


data["model_state_dict"] = model.state_dict()
data["model_type"] = model.__class__.__name__   #fixed 
# Save everything back to the same .pt file
torch.save(data, load_file)
print(f"Model saved to {load_file}")
