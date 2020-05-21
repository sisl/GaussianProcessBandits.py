import math
import numpy as np

from ridge_reg import ridge_reg
from sklearn.metrics import mean_squared_error
from validation import k_fold_cross_val

class ridge_reg_conversion: ## Rename it
    """
    Class converting the Ridge regression model class (ridge_reg) for the use
    in the black-box optimization.

    Attibutes:
    ---------------
    lambd : float
        L1 penalty coefficient.

    Methods:
    ------------
    decode(point)
        Decodes the parameter point (1,) from black-box optimization scale [0,1) to
        ridge_reg class scale [1e-15, 1e3)

    enocode()
        Encodes the parameter point from ridge_reg class scale [1e-15, 1e3) todo
        black-box optimization scale [0,1)

    train_and_test_cv(data, k = 10)

        Performs k-fold cross validation using the cross_val_score function with
        k folds.

    """
    def __init__(self, lambd = 0.5):

        #Initial model hyperparemetrs
        self.num_dims = 1
        self.lambd = lambd #or some other default value
        self.model = ridge_reg(self.lambd)

    def decode(self, point):
        """
        Decodes the parameter point (1,) from black-box optimization scale [0,1) to
        ridge_reg class scale [1e-15, 1e3)

        Args:
            point: (1,) contains the hyperparameters in the black-box optimization scale.
        """

        self.lambd = 10**(18 * point[0]- 15) #Equivalent to the lambda from 1e-15 to 1e3.

    def encode(self):
        """
        Encodes the parameter point from ridge_reg class scale [1e-15, 1e3) to
        black-box optimization scale [0,1)

        Returns:
            point: (1,)-sized numpy array with hyperparameters in the black-box optimization scale
        """

        #Converts hyperparameters value back to the points between [0,1]
        point = np.zeros((1,))
        point[0] = (math.log10(self.lambd) + 15)/18

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

        X = data.X
        y = data.y
        error = k_fold_cross_val(X, y, k, self.model)

        return error
