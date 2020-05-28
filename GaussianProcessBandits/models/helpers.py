import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

class Dataset:
    """
    Simple class for the datasets.
    """

    def __init__(self, X, y, train_loader = None, test_loader = None, dataloader = False):
        """
        Initializes a dataset for supervised learnining.

        Args:
            X: (n_samples,n_feature)-numpy array features with data
            y: (n_samples,)-numpy array target variable vector
        """
        self.X = X
        self.y = y
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dataloder = dataloader

def train_test_cv(data, folds, model):

    """
    Performs k-fold cross validation using the cross_val_score function with
    k folds.

    Uses the K-fold implementation from scikit learn.

    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    Args:
        X: (n_samples,n_feature)-numpy array features with data
        y: (n_samples,)-numpy array target variable vector
        folds: (int) number of folds the dataset is divided into int
        model: prediction model which contains fit and predict methods

    Returns:
        avg_error: mean generalization error generalization error over the
        folds

    """
    X = data.X
    y = data.y
    errors = []
    kf = KFold(n_splits = folds)

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        error = mean_squared_error(y_pred, y_test)
        errors.append(error)

    avg_error = np.mean(errors)

    return avg_error
