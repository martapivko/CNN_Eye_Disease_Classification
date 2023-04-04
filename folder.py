import numpy as np
from tqdm import tqdm



class KFolder:

    def __init__(self, k, x, y):

        self.k = k
        self.x = x
        self.y = y

        assert self.x.shape[0] == self.y.shape[0]

        self.folds = self._compute_folds()

    def _get_indicies(self, indice):

        valid_indice = indice

        if indice == 0:
            train_indicies = list(np.arange(indice+1, self.k))
        else:
            train_indicies = list()

            train_indicies.extend(list(np.arange(0, indice)))
            train_indicies.extend(list(np.arange(indice+1, self.k)))

        return train_indicies, valid_indice


    def _compute_folds(self):

        x_folds = np.split(self.x, self.k)
        y_folds = np.split(self.y, self.k)

        folds = list()
        
        print("Computing folds")
        for indice in tqdm(range(self.k)):

            x_train = list()
            y_train = list()
            x_valid = list()
            y_valid = list()

            train_indicies, valid_indice = self._get_indicies(indice)

            x_valid = x_folds[valid_indice]
            y_valid = y_folds[valid_indice]

            for train_indice in train_indicies:
                x_train.extend(x_folds[train_indice])
                y_train.extend(y_folds[train_indice])

            folds.append((np.array(x_train).astype(np.float32), np.array(y_train).astype(np.float32), np.array(x_valid).astype(np.float32), np.array(y_valid).astype(np.float32)))
        return folds
            

        


