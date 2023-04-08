import numpy as np
import torch


def get_batch_eof(data, batch_size, block_size, pad_token_int, non_eof_indices, cumsum_eof):
    """
    Return x and y
    """
    random_indices = non_eof_indices[np.random.randint(0, len(non_eof_indices), size=batch_size)]

    eof_indices = np.searchsorted(cumsum_eof, cumsum_eof[random_indices] + 1)

    x_subarrays = []
    y_subarrays = []
    for start, end in zip(random_indices, eof_indices):
        x_subarray = data[start:end + 1]
        y_subarray = data[start + 1:end + 1]

        x_subarray = np.pad(x_subarray, (0, max(0, block_size - len(x_subarray))), constant_values=pad_token_int)
        x_subarray = x_subarray[:block_size]

        y_subarray = np.pad(y_subarray, (0, max(0, block_size - len(y_subarray))), constant_values=pad_token_int)
        y_subarray = y_subarray[:block_size]

        x_subarrays.append(x_subarray)
        y_subarrays.append(y_subarray)

    x = np.stack(x_subarrays, axis=0)
    y = np.stack(y_subarrays, axis=0)

    return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))
