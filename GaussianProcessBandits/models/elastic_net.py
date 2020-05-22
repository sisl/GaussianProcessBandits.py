import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

class ElasticNetReg:
    """
    Elastic Net Class.
    Linear regression with combined L1 and L2 priors as regularizer.
    (Elastic Net). It uses scikit learn ElasticNet class.

    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

    """

    def __init__(self, lambda_1 = 0.5, lambda_2 = 0.25):
        """
        Initialize model parameters.
        Args:
            lambda_1 (float): L1 penalty coefficient.
            lambda_2 (float): L2 penalty coefficient.
        """

        #Initial model hyperparemetrs
        self.num_dims = 2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha = 2*self.lambda_2 + self.lambda_1
        self.l1_ratio = self.lambda_1/self.alpha
        self.max_iter = 1e06 ##Too many? It wll terminate early if it converges with tol=1e-04.
        self._model = ElasticNet(alpha = self.alpha, l1_ratio=self.l1_ratio, max_iter = self.max_iter)

    def fit(self,X,y):
        """
        Fits the parameters theta to the data matrix X (n_samples,n_features)
        and target variables y.

        Args:
            X (np.array): (n_samples,n_feature)-numpy array features with data
            y (np.array): (n_samples,)-numpy array target variable vector
        """

        self._model.fit(X,y)

    def predict(self,X):
        """
        Predicts the target_variable based on the provided data matrix X and
        learned parameters theta.

        Args:
            X (np.array): (n_samples,n_feature)-numpy array features with data

        Returns:
            y (np.array): (n_samples,)-numpy array with predicted target variable
        """

        y_pred = self._model.predict(X)

        return y_pred

    def decode(self, point):
        """
        Decodes the parameter point (2,) from black-box optimization scale [0,1) to
        ridge_reg class scale [1e-15, 1e3).

        Args:
            point (np.array): (2,)-sized numpy array with hyperparameters in the black-box optimization scale
        """


        self.lambda_1 = np.exp(12 * point[0]- 10) #Equivalent to the lambda from 1e-15 to 1e3.
        self.lambda_2 = np.exp(12 * point[1]- 10) #Equivalent to the lambda from 1e-15 to 1e3.

    def encode(self):
        """
        Encodes the parameter point from ridge_reg class scale [1e-15, 1e3) to
        black-box optimization scale [0,1).

        Returns:
            point (np.array): (2,)-sized numpy array with hyperparameters in the black-box optimization scale
        """

        #Converts hyperparameters value back to the points between [0,1]
        point = np.zeros((1,))
        point[0] = (np.log(self.lambda_1) + 10)/12
        point[1] = (np.log(self.lambda_2) + 10)/12

        #Add assert

        return point

    def train_test_cv(self, data, folds = 3):

        """
        Performs k-fold cross validation using the cross_val_score function with
        k folds.

        Uses the K-fold implementation from scikit learn.

        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

        Args:
            X (np.array): (n_samples,n_feature)-numpy array features with data
            y (np.array): (n_samples,)-numpy array target variable vector
            folds (int): number of folds the dataset is divided into int

        Returns:
            avg_error (float): mean generalization error generalization error over the
            folds

        """
        X = data.X
        y = data.y
        errors = []
        kf = KFold(n_splits = folds)

        for train_index, test_index in kf.split(X):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)

            error = mean_squared_error(y_pred, y_test)
            errors.append(error)

        avg_error = np.mean(errors)

        return avg_error
