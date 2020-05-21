import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from ridge_reg import ridge_reg
from elastic_net import elastic_net

def k_fold_cross_val(X, y, folds, model):

    """
    K-fold cross validation.
    Uses the K-fold implementation from scikit learn.

    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    Args:
        X: (n_samples,n_feature)-numpy array features with data
        y: (n_samples,)-numpy array target variable vector of shape (n_sample,)
        folds: (int) number of folds the dataset is divided into int
        model: prediction model which contains fit and predict methods

    Returns:
        avg_error: mean generalization error generalization error over the
        folds

    """

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
