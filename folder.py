import numpy as np
from tqdm import tqdm



class KFolder:

    def __init__(self, k, x, y):

        self.k = k
        self.x = x
        self.y = y

        assert self.x.shape[0] == self.y.shape[0]

        self.folds = self._compute_folds()

    def _get_indices(self, indice):

        valid_indice = indice

        if indice == 0:
            train_indices = list(np.arange(indice+1, self.k))
        else:
            train_indices = list()

            train_indices.extend(list(np.arange(0, indice)))
            train_indices.extend(list(np.arange(indice+1, self.k)))

        return train_indices, valid_indice


    def _compute_folds(self):
        num_samples = self.x.shape[0]
        fold_size = num_samples // self.k

        folds = []

        print("Computing folds")
        for i in tqdm(range(self.k)):
            start = i * fold_size
            end = start + fold_size

            x_valid = self.x[start:end]
            y_valid = self.y[start:end]

            train_indices, valid_indice = self._get_indices(i)

            x_train = np.concatenate([self.x[train_idx * fold_size : (train_idx + 1) * fold_size] for train_idx in train_indices])
            y_train = np.concatenate([self.y[train_idx * fold_size : (train_idx + 1) * fold_size] for train_idx in train_indices])

            folds.append((x_train.astype(np.float32), y_train.astype(np.float32), x_valid.astype(np.float32), y_valid.astype(np.float32)))

        return folds            