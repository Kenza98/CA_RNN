"""Feature extraction logic, isolated from the shared prep pipeline.

Each extractor maps (seq_block, target_map) -> (X_t, Y_t):
    seq_block  : (seq_length, H, W)
    target_map : (H, W)
    X_t        : (n_cells, seq_length, F)
    Y_t        : (n_cells, 1)

Both crop to the same interior cells, in row-major order, so cell i is the
same grid point across variants. NaN masking lives in prep.py.
"""


def extract_neighborhood(seq_block, target_map):
    """CA-style: 3x3 Moore neighborhood per cell. F = 9.
    Crop to interior cells: 1-cell border has no full neighborhood.
    \\TODO explore adding non-full neighborhoods to training as well?
    """
    seq_length = seq_block.shape[0]
    neigh = seq_block.unfold(1, 3, 1).unfold(2, 3, 1)   # (seq_length, H-2, W-2, 3, 3)

    #could have let torch infer with -1 but recuperating seq_length is more robust

    X_t = neigh.contiguous().view(seq_length, -1, 9).permute(1, 0, 2)
    Y_t = target_map[1:-1, 1:-1].contiguous().view(-1, 1)

    return X_t, Y_t


def extract_pointwise(seq_block, target_map):
    """Plain RNN: the cell's own history. F = 1.

    Cropped to the same interior cells as extract_neighborhood so the two
    variants are trained and evaluated on an identical cell set.
    """
    seq_length = seq_block.shape[0]
    centre = seq_block[:, 1:-1, 1:-1]
    X_t = centre.contiguous().view(seq_length, -1, 1).permute(1, 0, 2)
    Y_t = target_map[1:-1, 1:-1].contiguous().view(-1, 1)
    return X_t, Y_t


EXTRACTORS = {
    "nn": extract_pointwise,
    "nca": extract_neighborhood,
}