import torch

def evaluate_model(model, test_loader, device):
    """
    # This function returns the dictionary of results for a trained model checkpoint
    #
    """
    sample_se = []
    sample_ae = []
    preds = []

    model.eval()

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