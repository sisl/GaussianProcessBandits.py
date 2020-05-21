import numpy as np

class ridge_reg:

    """
    Class defining Linear Regression with L2 Penalty (Ridge Regression)


    Attibutes:
    ---------------
    lambd : float
        L1 penalty coefficient.

    Methods:
    ------------
    fit(X,y)
        Fits the parameters theta to the data matrix X (n_samples,n_features)
        and target variables y.

    predict(X)
        Predicts the target_variable based on the provided data matrix X and
        learned parameters theta.
    """
    def __init__(self, lambd = 0.5):
        """
        Parameters
        ----------
        lambd : float
            L1 penalty coefficient.
        theta : (n_features,1)-numpy array
            Parameters used in the Linear Regression model
        """

        #Initial model hyperparemetrs
        self.lambd = lambd
        self.theta = None

    def fit(self,X,y):
        """
        Fits the parameters theta to the data matrix X (n_samples,n_features)
        and target variables y.

        Args:
            X: (n_samples,n_feature)-numpy array features with data
            y: (n_samples,)-numpy array target variable vector of shape (n_sample,)
        """

        # Solve Ridge Regression analytically (lambfa*I + X.TX)^(-1)*X.Ty
        A = np.matmul(X.T,X)
        lambI = self.lambd * np.eye(A.shape[0])
        c = np.matmul(X.T,y)
        B = A + lambI
        self.theta = np.linalg.solve(B,c)
        self.theta = self.theta[:,np.newaxis]

    def predict(self,X):
        """
        Predicts the target_variable based on the provided data matrix X and
        learned parameters theta.

        Args:
            X: (n_samples,n_feature)-numpy array features with data

        Returns:
            y: (n_samples,)-numpy array with predicted target variable vector of shape
        """

        y_pred = np.matmul(X,self.theta)

        return y_pred
