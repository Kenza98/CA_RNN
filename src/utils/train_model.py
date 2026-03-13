import torch
import time


def train_model(model, optimizer, criterion, data_loader, nb_epochs=30, device="cpu"):
    N = len(data_loader)
    train_loss = []
    grad_history = {}
    for epoch in range(nb_epochs):
        print(f"Epoch {epoch+1}/{nb_epochs}\ncomputing ...\n...\n...")
        epoch_start = time.time()
        epoch_loss = 0.0
        for x_batch, y_batch in data_loader:
            # move x and y to GPU
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()  # reinitialize the gradients to avoid exploding (?)
            y_pred = model(x_batch)  # one forward step
            loss = criterion(y_pred, y_batch)
            loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if name not in grad_history:
                            grad_history[name] = []
                        grad_history[name].append(grad_norm)
            optimizer.step()
            epoch_loss += loss.item()  # .cpu() ???

        epoch_avg_loss = epoch_loss / N
        train_loss.append(epoch_avg_loss)
        epoch_time = time.time() - epoch_start
        # GPU memory usage
        if device == "cuda":
            mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
            print(f"GPU Mem: {mem_alloc:.1f}MB/{mem_reserved:.1f}MB | ")
        # Print epoch loss, time
        print(
            f"Loss: {epoch_avg_loss:.4f}\nTime: {epoch_time/60:.2f}min",
            flush=True,
        )
