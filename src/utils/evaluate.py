import torch

def evaluate_model(model, test_loader, device):
    """
    # This function returns the dictionary of results for a trained model checkpoint
    #
    """
    model = model.to(device)
    model.eval()

    sample_se = []
    sample_ae = []
    preds = []

    with torch.no_grad():
        for batch_seq, batch_tar in test_loader:
            batch_seq = batch_seq.to(device, non_blocking=True)
            batch_tar = batch_tar.to(device, non_blocking=True)
            outputs = model(batch_seq)
            sample_se.append(((outputs - batch_tar) ** 2).cpu())
            sample_ae.append(torch.abs(outputs - batch_tar).cpu())
            preds.append(outputs.cpu())

    se_tensor = torch.cat(sample_se)
    ae_tensor = torch.cat(sample_ae)
    pred_tensor = torch.cat(preds)

    mse = se_tensor.mean()
    mae = ae_tensor.mean()

    return {
        "mse": mse,
        "mae": mae,
        "squared_error": se_tensor.flatten(),
        "absolute_error": ae_tensor.flatten(),
        "prediction": pred_tensor.flatten()
    }


def get_baseline(data_loader, device):
    """this function computes a simple average of neighbors baseline"""
    neigh_indices = [i for i in range(9) if i != 4] #4 is indice of central pixel
    baselines = [] #instead of preds
    baseline_se = []
    baseline_ae = []
    for batch_seq, batch_tar in data_loader:
        #move to GPU
        batch_seq = batch_seq.to(device)
        batch_tar = batch_tar.to(device)
        
        #compute baseline as simple neighborhood average
        neighbors_only = batch_seq[:, -1, neigh_indices]
        batch_baseline = neighbors_only.mean(dim=1)
        baselines.append(batch_baseline)

        #compute absolute error
        batch_ae = torch.abs(batch_baseline - batch_tar.squeeze(-1))
        baseline_ae.append(batch_ae.cpu())

        #compute square error
        batch_se = torch.square(batch_baseline - batch_tar.squeeze(-1))
        baseline_se.append(batch_se.cpu())

    baseline_tensor = torch.cat(baselines)
    flat_baseline_tensor = baseline_tensor.flatten()
    se_tensor = torch.cat(baseline_se)
    ae_tensor = torch.cat(baseline_ae)

    mse = torch.mean(se_tensor)
    mae = torch.mean(ae_tensor)
    print(f"MSE has type {type(mse)}\n")
    print(f"MAE has type {type(mae)}\n")
    return {
        "mse" : mse, #baseline mse
        "mae" : mae, #baseline mae
        "squared_error" : se_tensor.flatten(),
        "absolute_error" : ae_tensor.flatten(),
        "baselines" : flat_baseline_tensor
    }

def quick_test_sanity(tmse, tmae, ae_tensor, se_tensor):
    if isinstance(tmse, torch.Tensor):
        mse = tmse.item()
    if isinstance(tmae, torch.Tensor):
        mae = tmae.item()
    # === SHOWING QUANTILES ===
    quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    
    ae_q = torch.quantile(ae_tensor, quantiles)
    se_q = torch.quantile(se_tensor, quantiles)

    print("\n***Absolute Error Quantiles***")
    for q, v in zip(quantiles, ae_q):
        print(f"{q.item():>5.2f} : {v.item():.6f}")

    print("\n***Squared Error Quantiles***")
    for q, v in zip(quantiles, se_q):
        print(f"{q.item():>5.2f} : {v.item():.6f}")
def show_quartiles(ae_tensor, se_tensor):
    # computes quartiles and returns them + shows them
    quartiles = torch.tensor([0.25, 0.5, 0.75])
    print(torch.quantile(ae_tensor.flatten(), quartiles))
    for v in quartiles:
        print(f"{v.item():.6f}")

























