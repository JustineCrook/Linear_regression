import numpy as np


###############################################################################################################################
## HELPER METHOD



def dict_samples_to_array(samples):
    """Convert a dictionary of samples to a 2-dimensional array."""
    data = []
    names = []

    for key, x in samples.items():
        if x.ndim == 1:
            data.append(x)
            names.append(key)
        elif x.ndim == 2:
            for i in range(x.shape[-1]):
                data.append(x[:, i])
                names.append(f"{key}_{i}")
        elif x.ndim == 3:
            for i in range(x.shape[-1]):
                for j in range(x.shape[-2]):
                    data.append(x[:, j, i])
                    names.append(f"{key}_{j}_{i}")
        else:
            raise ValueError("Invalid dimensionality of samples to stack.")

    return np.vstack(data).T, names