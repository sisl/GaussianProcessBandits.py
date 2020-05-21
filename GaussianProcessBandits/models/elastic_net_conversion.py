import math
import numpy as np

from elastic_net import elastic_net
from sklearn.metrics import mean_squared_error
from validation import k_fold_cross_val

class elastic_net_conversion: ## Rename it
    """
    Class converting the Elastic Net model class (elastic_net) for the use
    in the black-box optimization.

    Attibutes:
    ---------------
    lambda_1 : float
        L1 penalty coefficient.

    lambda_2 : float
        L2 penalty coefficient.

    Methods:
    ------------
    decode(point)
        Decodes the parameter point (2,) from black-box optimization scale [0,1) to
        ridge_reg class scale [1e-15, 1e3)

    enocode()
        Encodes the parameter point from ridge_reg class scale [1e-15, 1e3) to
        black-box optimization scale [0,1)

    train_and_test_cv(data, k = 10)

        Perform k-fold cross validation using the cross_val_score function with
        k folds .

    """
    def __init__(self, lambda_1 = 0.5, lambda_2 = 0.5):

        """
        Parameters
        ----------
        num_dims : int
            number of dimensions of the parameters.
        lambda_1 : float
            L1 penalty coefficient.
        lambda_2 : float
            L2 penalty coefficient.
        model : Elastic Net model type
            Elastic Net model defined in ridge_reg.py
        """

        #Initial model hyperparemetrs
        self.num_dims = 2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.model = elastic_net(self.lambda_1, self.lambda_2)

    def decode(self, point):
        """
        Decodes the parameter point (2,) from black-box optimization scale [0,1) to
        ridge_reg class scale [1e-15, 1e3)

        Args:
            point: (2,)-sized numpy array with hyperparameters in the black-box optimization scale
        """


        self.lambda_1 = 10**(18 * point[0]- 15) #Equivalent to the lambda from 1e-15 to 1e3.
        self.lambda_2 = 10**(18 * point[1]- 15) #Equivalent to the lambda from 1e-15 to 1e3.

    def encode(self):
        """
        Encodes the parameter point from ridge_reg class scale [1e-15, 1e3) to
        black-box optimization scale [0,1)

        Returns:
            point: (2,)-sized numpy array with hyperparameters in the black-box optimization scale
        """

        #Converts hyperparameters value back to the points between [0,1]
        point = np.zeros((1,))
        point[0] = (math.log10(self.lambda_1) + 15)/18
        point[1] = (math.log10(self.lambda_2) + 15)/18

        #Add assert

        return point

    def train_and_test_cv(self, data, k = 10):
        """
        Performs k-fold cross validation using the cross_val_score function with
        k folds.

        Args:
            data: data-type containting the dataset
            k: (int) number of folds in the cross validation
        """

        error = k_fold_cross_val(X, y, k, self.model)

        return error
