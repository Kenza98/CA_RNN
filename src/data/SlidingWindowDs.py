import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr


class SlidingWindowDs(Dataset):

    def __init__(self, xr_data, seq_length=8):
        # kernel_size=3
        """
        xr_data: xarray.DataArray with dims ('time', 'lat', 'lon', 'feature')
        \\TODO make it Dask-backed for lazy loading.
        """
        self.data = xr_data
        self.seq_length = seq_length
        self.global_mean = float(self.data.mean().compute())  # lazy compute once

        # Extract dimensions
        self.T = self.data.sizes["time"]
        self.H = self.data.sizes["latitude"]
        self.W = self.data.sizes["longitude"]
        # self.F = self.data.sizes["thetao"]

    def __len__(self):  # nb of samples in the xr dataset
        return (self.T - self.seq_length - 1) * self.H * self.W

    def __getitem__(self, index):
        # compute the T, H, W indices from flat index

        t_idx = index // (self.H * self.W)
        h_idx = index % (self.H * self.W) // self.W  # vertical coordinate
        w_idx = index % (self.H * self.W) % self.W  # horizontal coordinate

        # to avoid being out of range, raise IndexError if the indices are out of bounds
        # Skip borders: not enough space for neighborhood
        if h_idx < 1 or h_idx >= self.H - 1 or w_idx < 1 or w_idx >= self.W - 1:
            # raise IndexError("Index corresponds to a border pixel, skipping.")
            # Shape: (seq_length, 3, 3
            seq = np.full((self.seq_length, 3, 3), self.global_mean, dtype=np.float32)
            t_slice = slice(t_idx, t_idx + self.seq_length)
            h_min = max(0, h_idx - 1)
            h_max = min(self.H, h_idx + 2)
            w_min = max(0, w_idx - 1)
            w_max = min(self.W, w_idx + 2)

            # Extract valid sub-patch
            valid_patch = self.data.isel(
                time=t_slice,
                latitude=slice(h_min, h_max),
                longitude=slice(w_min, w_max),
            ).values.astype(np.float32)
            # Determine where to insert it inside the 3x3 patch
            insert_h_start = 1 - (h_idx - h_min)
            insert_h_end = insert_h_start + valid_patch.shape[1]
            insert_w_start = 1 - (w_idx - w_min)
            insert_w_end = insert_w_start + valid_patch.shape[2]

            seq[:, insert_h_start:insert_h_end, insert_w_start:insert_w_end] = (
                valid_patch
            )
            seq = seq.reshape(self.seq_length, -1)

            # Create target (center pixel at final time)
            target_val = self.data.isel(
                time=t_idx + self.seq_length, latitude=h_idx, longitude=w_idx
            ).values.astype(np.float32)

            return torch.tensor(seq), torch.tensor([target_val])
        # Input time slice
        t_slice = slice(t_idx, t_idx + self.seq_length)

        # Spatial neighborhood slice
        h_slice = slice(h_idx - 1, h_idx + 2)
        w_slice = slice(w_idx - 1, w_idx + 2)

        # Lazy slice from xarray (Dask-backed)
        neighborhood = self.data.isel(time=t_slice, latitude=h_slice, longitude=w_slice)

        # Convert to torch tensor -> ici quand j'appelle .values, c'est là que ça va load en mémoire pour la première fois
        neighborhood = torch.tensor(neighborhood.values, dtype=torch.float32)

        # Flatten spatial dims into one dimension: (seq_length, neighborhood_size * nb_features)
        neighborhood = neighborhood.reshape(self.seq_length, -1)

        # Target = center pixel at t+seq_length
        target_time = t_idx + self.seq_length
        target = self.data.isel(time=target_time, latitude=h_idx, longitude=w_idx)
        target = torch.tensor(target.values, dtype=torch.float32)

        return neighborhood, target

    def __str__(self):
        return f"""This Dataset has :\n{self.T} timestamps\n
        {self.W} * {self.H} grid size\n{self.F} features\n"""
