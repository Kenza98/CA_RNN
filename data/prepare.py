import h5py
import torch
from torch.utils.data import random_split
import numpy as np
import os
from SlidingWindowDs import *


# \\TODO [DONE] check if it works to wrap my sw into TensorDataset and save it to .pt
def create_linear_dataset(input_file, dataset_name, output_file):
    """
    Prepare data for learning linear dynamics:
    - Loads data from .h5 \\TODO or takes a ready-made tensor with xarray
    - Applies sliding window (seq_length=1)
    - Converts to raw tensor form
    - Saves to .pt file
    """
    with h5py.File(input_file, "r") as f:
        dataset = f[dataset_name][:]
    dataset = dataset[..., np.newaxis]  # add a dimension for the features F = 1
    full_dataset = SlidingWindowDs(dataset, 1)
    N = len(full_dataset)
    train_size = int(0.8 * N)
    test_size = N - train_size
    lengths = [train_size, test_size]
    train_dataset, test_dataset = random_split(full_dataset, lengths)
    # extract x,y pairs
    X_TRAIN, Y_TRAIN = zip(*[train_dataset[i] for i in range(len(train_dataset))])
    X_TEST, Y_TEST = zip(*[test_dataset[i] for i in range(len(test_dataset))])

    X_TEST = torch.stack(X_TEST)
    Y_TEST = torch.stack(Y_TEST)
    X_TRAIN = torch.stack(X_TRAIN)
    Y_TRAIN = torch.stack(Y_TRAIN)

    # save to .pt file
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    pt_file = torch.load(output_file)
    pt_file["X_TRAIN"] = X_TRAIN
    pt_file["Y_TRAIN"] = Y_TRAIN
    pt_file["X_TEST"] = X_TEST
    pt_file["Y_TEST"] = Y_TEST
    torch.save(pt_file, output_file)
    print(
        f"Created sliding window dataset with train size {len(train_dataset)} and test size {len(test_dataset)} samples."
    )


def create_window_dataset(input_file, dataset_name, output_file, seq_length):
    """
    Prepare data for learning timeseries dynamics using sliding window method:
    - Loads data from .h5
    - Applies sliding window
    - Converts to raw tensor form
    - Saves to .pt file
    """
    with h5py.File(input_file, "r") as f:
        dataset = f[dataset_name][:]
    dataset = dataset[..., np.newaxis]  # add a dimension for the features F = 1
    full_dataset = SlidingWindowDs(dataset, seq_length)
    N = len(full_dataset)
    train_size = int(0.8 * N)
    test_size = N - train_size
    lengths = [train_size, test_size]
    train_dataset, test_dataset = random_split(full_dataset, lengths)

    # extract x,y pairs
    X_TRAIN, Y_TRAIN = zip(*[train_dataset[i] for i in range(len(train_dataset))])
    X_TEST, Y_TEST = zip(*[test_dataset[i] for i in range(len(test_dataset))])

    X_TEST = torch.stack(X_TEST)
    Y_TEST = torch.stack(Y_TEST)
    X_TRAIN = torch.stack(X_TRAIN)
    Y_TRAIN = torch.stack(Y_TRAIN)

    # save to EXISTING .pt file
    try:
        pt_file = torch.load(output_file)
        print("file loaded successfully")
        pt_file["X_TRAIN"] = X_TRAIN
        pt_file["Y_TRAIN"] = Y_TRAIN
        pt_file["X_TEST"] = X_TEST
        pt_file["Y_TEST"] = Y_TEST
        torch.save(pt_file, output_file)

    except FileNotFoundError:
        """
        THIS APPROACH IS FOR FIRST TIME CREATING PT FILE

        """
        print("file not found, creating new one")
        data_to_save = {
            "X_TRAIN": X_TRAIN,
            "Y_TRAIN": Y_TRAIN,
            "X_TEST": X_TEST,
            "Y_TEST": Y_TEST,
        }

        torch.save(data_to_save, output_file)
    print(
        f"Created sliding window dataset with train size {len(train_dataset)} and test size {len(test_dataset)} samples."
    )


def main():
    dataset_name = "wca_exp_02"
    h5_file = "src/fvg_adriatic_data/twoCA.h5"
    # ca05_sli_win
    output_file = "synthetic_data_interpolation/data/ca02_lr.pt"
    create_window_dataset(h5_file, dataset_name, output_file, 1)

    # output_file = "synthetic_data_interpolation/data/ca_exp_data.pt"
    # create_linear_dataset(h5_file, dataset_name, output_file)


if __name__ == "__main__":
    main()
