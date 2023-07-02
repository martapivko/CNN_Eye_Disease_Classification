import random
import numpy as np


def shuffle_data(data_imgs: list, data_labels: list):
    temp = list(zip(data_imgs, data_labels))
    random.shuffle(temp)
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    res1, res2 = list(res1), list(res2)
    return np.array(res1), np.array(res2)