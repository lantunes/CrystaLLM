
def array_split(arr, num_splits):
    split_size, remainder = divmod(len(arr), num_splits)
    splits = []
    start = 0
    for i in range(num_splits):
        end = start + split_size + (i < remainder)
        splits.append(arr[start:end])
        start = end
    return splits
