import numpy as np


def is_above(x, y, dir):
    return np.dot(dir, x - y) > 0


def counting_sort(arr, key_id):
    """
    :param arr: list of tuples
    :param key_id: id of a tuple item to sort by
    :return:
    """
    n = len(arr)
    out = [0] * n
    bin_n = n // 2
    bins = [0] * bin_n

    # count how many array items each bin contains by key
    for i in range(n):
        key = arr[i][key_id]
        bins[key % bin_n] += 1

    for i in range(1, bin_n):
        bins[i] += bins[i-1]

    for i in range(n):
        key = arr[i][key_id]
        out[n - bins[key % bin_n]] = arr[i]
        bins[key % bin_n] -= 1
    return out
