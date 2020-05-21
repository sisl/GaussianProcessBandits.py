#import random
import numpy as np

def gpbandits(model, data, iters=10, kernel='se', cl=0.1, v=0.0, num_samples=500):
    """
    Chooses the best model by running the Gaussian Process Bandits algorithm on the hyperparameter space with the expected
    improvement heuristic.

    Args:
        model: a data model, which must include a num_dims field for the number of hyperparameters, implement 
            a ``decode`` function which sets hyperparameters, an ``encode`` function which maps model hyperparameters 
            back to (0,1)^num_dims, and a ``train_test_cv`` function which returns a cross-validated model score to be 
            minimized.
        data: the raw training data (in appropriate form) to be fed to model.train_test_cv
        iters (int): the number of iterations to run the bandits algorithm
        kernel (string): the kernel function to use in the Gaussian Process (default 'se' for squared exponential)
        cl (float): the characteristic length scale for the kernel
        v (float): the noise variance for the Gaussian Process
        num_samples (int): the number of samples
    Returns:
        model: a data model with the best hyperparameters set
    """
    
    num_dims = model.num_dims # number of hyperparater dimensions
    points = []
    scores = []
    
    # initial model evaluation
    points.append(model.encode())
    scores.append(model.train_test_cv(data))
    
    # loop
    for i in range(iters):
    
        # sample m random points from [0,1]^num_dims
        candidates = sample(num_dims, num_samples)
        
        # finding GP posterior
        
        K = formK(points, candidates, kernel, cl, v)
        
        
        # choose new point with best expected improvement
        
        # set hyperparameters with best sampled point
        model.decode(best_point) 
        
        # return re-encoded point
        new_point = model.encode() 
        
        # evaluate model
        new_score = model.train_test_cv(data) 
        
        # append to points/scores lists
        points.append(model.encode()) 
        scores.append(model.train_test_cv(data)) 
        
        # save progress
        save(points, scores)
    
    # return best model
    s = 0
    best_overall_point = 0
    model.decode(best_overall_point)
    return model
        
        
def formK(x, y, kernel, cl):
    """
    Form a kernel matrix
    Args:
        x (np.array): (nx, num_dims)-sized numpy array for first set of points
        y (np.array): (ny, num_dims)-sized numpy array for second set of points
        kernel (string): the kernel function to use in the Gaussian Process (default 'se' for squared exponential)
        cl (float): the characteristic length scale for the kernel
    Returns:
        K (np.array): (nx, ny)-sized numpy array Kernel matrix
    """
    
    if kernel == 'se':
        k = lambda x,y: np.exp(-np.sum((x-y)**2)/2/cl**2)
    else:
        raise('Kernel %s not implemented' %(kernel))
    
    # form kernel matrix
    K = np.zeros(len(x),len(y))
    for i in range(len(x)):
        for j in range(len(y)):
            K[i,j] = k(x[i],y[j]) 
            
    return K
    
def sample(num_dims, num_samples):
    """
    Sample num_samples samples in [0,1)^num_dims
    Args:
        num_dims (int): number of hyperparameters / dimensions to the Gaussian Process
        num_Samples (int): number of samples to evaluate at each iteration of the Gaussian Process Bandits algorithm
    Returns:
        samples (np.array): (num_samples, num_dims)-sized numpy array of samples
        
    """
    samples = np.random.rand(num_samples, num_dims)
    ### TODO: Update with a uniform sampling plan to fill space 
    return samples

def save(points, scores):
    """
    Save progress so far as a csv file, with each line containing each encoded point and associated model score.
    Args:
        points (np.array): (num_points, num_dims)-sized array of points evaluated so far
        scores (np.array): (num_points)-sized array of scores from evaluated points
    """
    pass