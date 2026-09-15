def normalize(X, Y, mean, std):
    return (X - mean) / std, (Y - mean) / std
