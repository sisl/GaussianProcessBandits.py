import numpy as np
import math

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

class RidgeReg:

    """
    Class defining Linear Regression with L2 Penalty (Ridge Regression)
    """
    def __init__(self, lambd = 100.0):
        """
        Initializes model parameters.
        Args:
            lambd (float): L1 penalty coefficient.
        """
        #Initial model hyperparemetrs
        self.lambd = lambd
        self.num_dims = 1
        self.theta = None

    def fit(self,X,y):
        """
        Fits the parameters theta to the data matrix X (n_samples,n_features)
        and target variables y.

        Args:
            X (np.array): (n_samples,n_feature)-numpy array features with data
            y (np.array): (n_samples,)-numpy array target variable vector
        """

        # Solve Ridge Regression analytically (lambfa*I + X.TX)^(-1)*X.Ty
        X_ = np.hstack((X,np.ones((X.shape[0],1))))
        A = np.matmul(X_.T,X_)
        lambI = self.lambd * np.eye(A.shape[0])
        c = np.matmul(X_.T,y)
        B = A + lambI
        self.theta = np.linalg.solve(B,c)
        self.theta = self.theta[:,np.newaxis]

    def predict(self,X):
        """
        Predicts the target_variable based on the provided data matrix X and
        learned parameters theta.

        Args:
            X (np.array): (n_samples,n_feature)-numpy array features with data

        Returns:
            y (np.array): (n_samples,)-numpy array with predicted target variable
        """
        X_ = np.hstack((X,np.ones((X.shape[0],1))))
        y_pred = np.matmul(X_,self.theta)

        return y_pred

    def decode(self, point):
        """
        Decodes the parameter point (2,) from black-box optimization scale [0,1) to
        ridge_reg class scale [1e-15, 1e3).

        Args:
            point (np.array): (1,)-sized numpy array with hyperparameters in the black-box optimization scale
        """

        self.lambd = np.exp(12 * point[0]- 10) #Equivalent to the lambda from 1e-15 to 1e3.
        #print(self.lambd)

    def encode(self):
        """
        Encodes the parameter point from ridge_reg class scale [1e-15, 1e3) to
        black-box optimization scale [0,1).

        Returns:
            point (np.array): (1,)-sized numpy array with hyperparameters in the black-box optimization scale
        """

        #Converts hyperparameters value back to the points between [0,1]
        point = np.zeros((1,))
        point[0] = (np.log(self.lambd) + 10)/12

        #Add assert

        return point

    def train_test_cv(self, data, folds = 4):
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
