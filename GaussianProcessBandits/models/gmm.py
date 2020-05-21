from sklearn import mixture
import numpy as np

class GMM:
    """
    Class to support Gaussian Mixture Model fitting for unsupervised learning
    """
    
    def __init__(self, k_init=2, covariance_type='full', num_fits=20, max_clusters=20):
        """
        Initialize model parameters
        Args:
            k_init (int): initial number of gaussians in mixture (default: 10)
            covariance_type (string): GMM covariance type (default: 'full')
            num_fits (int): number of times to fit random holdout sets during cross validation
            max_clusters (int): maximum number of clusters to for GP bandit to try
        """
        self.num_dims = 1
        self.k = k_init
        self.covariance_type = covariance_type
        self.num_fits = num_fits
        self.max_clusters = max_clusters
        
    def decode(self, point):
        """
        Set hyperparameters based on decoding scheme of point
        
        Args:
            point (np.array): (num_dims,)-sized numpy array of encoded hyperparameters in [0,1)
        """
        self.k = int(np.ceil(self.max_clusters*point[0]))
    
    def encode(self):
        """
        Return model hyperparameters encoded into [0,1)^num_dims
        
        Returns:
           point (np.array): (num_dims,)-sized numpy array of encoded hyperparameters in [0,1)
        """
        point = np.array([(self.k-1)/self.max_clusters])
        return point
        
    def train_test_cv(self, data):
        """
        Return model score (to be minimized) with data data using set hyperparameters
        
        Args:
            data (np.array): (num_points, num_features)-sized numpy array data stream to fit and score a GMM
        Returns:
            score (float): the score of the fit model (to be minimized)
        """
        
        # fit k gaussians to a random 70% of training data
        # calcualte average nll on 30% heldout data
        # repeat num_fits times
        
        n, m = data.shape
        nll = np.zeros(self.num_fits)
        for i in range(self.num_fits):
            idxs = np.random.permutation(n)
            train_idxs = idxs[:int(n*.7)]
            test_idxs = idxs[int(n*.7):]
            gmm = mixture.GaussianMixture(n_components=self.k, covariance_type=self.covariance_type)
            gmm.fit(data[train_idxs,:])
            meanll = gmm.score(data[test_idxs])
            nll[i] = -meanll            
            
        # return average holdout nll
        score = nll.mean()
        return score
            
            