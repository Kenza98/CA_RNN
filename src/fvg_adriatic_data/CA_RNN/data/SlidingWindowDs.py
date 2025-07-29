import torch
from torch.utils.data import Dataset

# for a quick test :
import h5py


class SlidingWindowDs(Dataset):

    def __init__(self, data, seq_length=6):
        """
        data: np.array of shape (T, H, W, F) where F = 1 for now.
        seq_length: number of timesteps to use as input, will be = 8.
        kernel_size: spatial neighborhood size (must be odd, for now = 3)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length
        self.kernel_size = 3
        # assert kernel_size % 2 == 1,"kernel_size must be odd"

        self.pad = self.kernel_size // 2  # 1 for now
        self.H, self.W, self.T, self.F = self.data.shape
        p4d = (0, 0, 0, 0, 2, 2, 2, 2)
        self.padded_data = torch.nn.functional.pad(self.data, p4d, "constant", 1)

    def __len__(self):  # nb of samples in the dataseth5py
        return (self.T - self.seq_length - 1) * self.H * self.W

    def __getitem__(self, index):

        t_idx = index // (self.H * self.W)
        h_idx = index % (self.H * self.W) // self.W  # vertical coordinate
        w_idx = index % (self.H * self.W) % self.W  # horizontal coordinate

        # to avoid being out of range, add the padding to the indexes
        h_idx += self.pad
        w_idx += self.pad

        # print(f"h_{index}: {h_idx-1} , {h_idx} , {h_idx+1} \n w_{index} {w_idx-1} , {w_idx} , {w_idx+1}")

        # Extract neighborhood around (h_idx, w_idx)
        neighborhood = self.padded_data[
            h_idx - self.pad : h_idx + self.pad + 1,
            w_idx - self.pad : w_idx + self.pad + 1,
            t_idx : t_idx + self.seq_length,
            :,  # features, for now only 1.
        ]  # Shape: (seq_len, K, K, F)

        # Target could be: center cell at t+seq_length (prev t+1)

        neighborhood = neighborhood.permute(2, 0, 1, 3)
        # print(neighborhood.shape)
        # Shape: (seq_len, K, K, F) -> (seq_len, K*K, F)

        neighborhood = neighborhood.reshape(self.seq_length, -1)

        target = self.padded_data[
            h_idx, w_idx, t_idx + self.seq_length, :
        ]  # Shape: (F,)

        return neighborhood, target

    def __str__(self):
        return f"""This Dataset has :\n{self.T} timestamps\n
        {self.W} * {self.H} grid size\n{self.F} features\n"""


def main():
    array = torch.zeros((500, 500, 10, 1), dtype=torch.float32)

    h5_file = "src/fvg_adriatic_data/twoCA.h5"

    with h5py.File(h5_file, "r") as f:
        dataset = f["wca_exp_02"][:, :, :10]

    dataset = dataset[..., torch.newaxis]  # add a dimension for the features bc = 1

    dataset = SlidingWindowDs(dataset, seq_length=8)

    seq, tar = dataset[0]
    print(f"seq shape: {seq.shape}")
    print(f"target shape: {tar.shape}")


if __name__ == main():
    main()


"""
def visualize_neighborhood_range(
    t_idx, h_idx, w_idx, pad, kernel_size, seq_length, data_shape
):
    
    # Calculate the start and end indices of the patch
    half_k = kernel_size // 2

    # Print the neighborhood coordinates for a specific time step and cell
    print(f"Time step: {t_idx}")
    print(f"Center: ({h_idx}, {w_idx})")

    # Time indices (seq_len)
    t_range = list(range(t_idx, t_idx + seq_length))

    # Spatial ranges for the current neighborhood centered at (h_idx, w_idx)
    h_range = list(range(h_idx - pad, h_idx + pad + 1))
    w_range = list(range(w_idx - pad, w_idx + pad + 1))

    print(f"Time indices (t_idx to t_idx + seq_length): {t_range}")
    print(f"Spatial indices (h_idx - pad to h_idx + pad): {h_range}")
    print(f"Spatial indices (w_idx - pad to w_idx + pad): {w_range}")

    # Visualize the shape and range of the extracted neighborhood
    print(f"Shape of neighborhood: ({seq_length}, {kernel_size}, {kernel_size}, F)")

    # Ensure the ranges are within the bounds of the data shape (H, W)
    h_range = [h for h in h_range if 0 <= h < data_shape[1]]
    w_range = [w for w in w_range if 0 <= w < data_shape[2]]

    print(f"Valid h_range: {h_range}")
    print(f"Valid w_range: {w_range}")

"""
