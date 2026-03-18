import time
import torch
    
def train_model(model, train_loader, optimizer, criterion, num_epochs, device):
    """Generic training loop for RNN-based models."""
    
    N = len(train_loader.dataset)
    train_loss = []
    grad_history = {}
    
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}\ncomputing ...\n...\n...")
        epoch_start = time.time()
        epoch_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
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
            epoch_loss += loss.item()
        
        epoch_avg_loss = epoch_loss / N
        train_loss.append(epoch_avg_loss)
        epoch_time = time.time() - epoch_start
        
        if device.type == "cuda":
            mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
            print(f"GPU Mem: {mem_alloc:.1f}MB/{mem_reserved:.1f}MB")
        
        print(f"Loss: {epoch_avg_loss:.4f}\nTime: {epoch_time/60:.2f}min", flush=True)
    
    return train_loss, grad_history