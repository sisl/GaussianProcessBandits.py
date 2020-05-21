import numpy as np

from sklearn.linear_model import ElasticNet

class elastic_net:
    """

    Linear regression with combined L1 and L2 priors as regularizer
    (Elastic Net). It uses scikit learn Elastic Net class.

    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

    Attibutes:
    ---------------
    lambda_1 : float
        L1 penalty coefficient.
    lambda_2 : float
        L2 penalty coefficient.

    Methods:
    ------------
    fit(X,y)
        Fits the parameters theta to the data matrix X (n_samples,n_features)
        and target variables y. It uses scikit learn implemented ElasticNet class and gradient descent as
            an optimization method.

    predict(X)
        Predicts the target_variable based on the provided data matrix X and
        learned parameters theta.
    """

    def __init__(self, lambda_1 = 0.5, lambda_2 = 0.25):
        """
        Parameters
        ----------
        lambda_1 : float
            L1 penalty coefficient.
        lambda_2 : float
            L2 penalty coefficient.
        l1_ratio : float
            (for the sklearn package). The ElasticNet mixing parameter.
            It can be derived from lambda_1 and lambda_2.
        alpha : float
            (For the sklearn package). Constant that multiplies the penalty terms.
            It can be derived from lambda_1 and lambda_2.
        """

        #Initial model hyperparemetrs
        self.alpha = 2*lambda_2 + lambda_1
        self.l1_ratio = lambda_1/self.alpha
        self.max_iter = 1e06 ##Too many? It wll terminate early if it converges with tol=1e-04.
        self._model = ElasticNet(alpha = self.alpha, l1_ratio=self.l1_ratio, max_iter = self.max_iter)

    def fit(self,X,y):
        """
        Fits the parameters theta to the data matrix X (n_samples,n_features)
        and target variables y.

        Args:
            X: (n_samples,n_feature)-numpy array features with data
            y: (n_samples,)-numpy array target variable vector of shape (n_sample,)
        """

        self._model.fit(X,y)

    def predict(self,X):
        """
        Predicts the target_variable based on the provided data matrix X and
        learned parameters theta.

        Args:
            X: (n_samples,n_feature)-numpy array features with data

        Returns:
            y: (n_samples,)-numpy array with predicted target variable vector of shape
        """

        y_pred = self._model.predict(X)

        return y_pred
