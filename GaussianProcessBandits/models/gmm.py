import sklearn as skl
import numpy as np

class GMM:
    """
    Class to support Gaussian Mixture Model fitting for unsupervised learning
    """
    
    def self.__init__():
        """
        Initialize model parameters
        """
        self.num_dims = 1
        self.k = 10
        self.covariace_type = 'full'
        
        
        
    def self.decode(point):
        """
        Set hyperparameters based on decoding scheme of point
        Args:
            point (np.array): (num_dims,)-sized numpy array of encoded hyperparameters in [0,1)
        """
        self.k = int(np.ceil(20*point[0]))
    
    def self.encode():
        """
        Return model hyperparameters encoded into [0,1)^num_dims
        Returns:
           point (np.array): (num_dims,)-sized numpy array of encoded hyperparameters in [0,1)
        """
        point = np.array([(self.k-1)/20])