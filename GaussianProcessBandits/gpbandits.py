import numpy as np
from scipy.stats import norm

def gpbandits(model, data, iters=10, kernel='se', cl=0.1, v=0.0, num_samples=500, verbose=True, best_model_log=False):
    """
    Chooses the best model by running the Gaussian Process Bandits algorithm on the hyperparameter space with the expected
    improvement heuristic.

    Args:
        model: a data model, which must include a num_dims field for the number of hyperparameters, implement
            a ``decode`` function which sets hyperparameters, an ``encode`` function which maps model hyperparameters
            back to [0,1)^num_dims, and a ``train_test_cv`` function which returns a cross-validated model score to be
            minimized.
        data: the raw training data (in appropriate form) to be fed to model.train_test_cv
        iters (int): the number of iterations to run the bandits algorithm
        kernel (string): the kernel function to use in the Gaussian Process (default 'se' for squared exponential)
        cl (float): the characteristic length scale for the kernel
        v (float): the noise variance for the Gaussian Process
        num_samples (int): the number of samples
        verbose (bool): if True, print each point and score
        best_model (bool): if True, return list of best models (points) after each iteration
    Returns:
        model: a data model with the best hyperparameters set
    """

    num_dims = model.num_dims # number of hyperparameter dimensions

    # initial model evaluation
    points = model.encode()[np.newaxis,:]
    scores = np.array([model.train_test_cv(data)])

    # best model and corresponding value at each iteration
    if best_model_log:
        best_point_tmp = []
        best_point_tmp.append(points[0,:])

    # print update
    if verbose:
        print("Iteration: %03d | Score: %.06e" %(0, scores[0]))
        #print("Iteration: %03d | Design Point: %f | Score: %.06e" %(0,points[0,:] scores[0]))

    # loop
    for i in range(iters):

        # sample num_Samples random points from [0,1)^num_dims
        candidates = sample(num_dims, num_samples)

        # find GP posterior
        A = formK(candidates, candidates, kernel, cl)
        B = formK(points, points, kernel, cl) + v*np.eye(points.shape[0])
        C = formK(candidates, points, kernel, cl)
        tmp = C.dot(np.linalg.inv(B))
        mu = tmp.dot(scores)
        Sigma = A - tmp.dot(C.T)
        var = np.diagonal(Sigma) + np.finfo(float).eps
        sig = np.sqrt(var)

        # choose new point with best expected improvement
        exp_imp = expected_improvement(scores.min(), mu, sig)
        best_idx = np.argmax(exp_imp)
        best_point = candidates[best_idx]

        # set hyperparameters with best sampled point
        model.decode(best_point)

        # return re-encoded point
        new_point = model.encode()

        # evaluate model
        new_score = model.train_test_cv(data)

        # append to points/scores lists
        points = np.vstack((points, best_point)) # use best_point, not re-encoded new_point to break discrete symmetries
        scores = np.append(scores, new_score)

        # save progress
        save_checkpoint(points, scores)

        # print update
        if verbose:
            print("Iteration: %03d | Score: %.06e" %(i+1, new_score))
            #print("Iteration: %03d | Design Point: %f | Score: %.06e" %(i+1, best_point, new_score))

        if best_model_log:
            ind = np.argmin(scores)
            best_point_tmp.append(points[ind])



    # return best model
    ind = np.argmin(scores)
    best_overall_point = points[ind]
    model.decode(best_overall_point)

    if not best_model_log:
        return model
    else:
        return model, best_point_tmp


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
    K = np.zeros((x.shape[0], y.shape[0]))
    for i in range(len(x)):
        for j in range(len(y)):
            K[i,j] = k(x[i], y[j])

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

def expected_improvement(ymin, mu, sig):
    """
    Return the expected improvement of each candidate point
    Args:
        ymin (float): the best score so far
        mu (np.array): (num_samples,)-sized numpy array of posterior mean values at each candidate sample
        sig (np.array): (num_samples,)-sized numpy array of posterior stdev values at each candidate samples
    Returns:
        ei (np.array): (num_samples,)-sized numpy array of expected improvement at each sample

    """
    p_imp = norm.cdf((ymin-mu)/sig)
    p_ymin = norm.pdf((ymin-mu)/sig)
    ei = (ymin-mu)*p_imp + sig*p_ymin
    return ei

def save_checkpoint(points, scores):
    """
    Save progress so far as a csv file, with each line containing each encoded point and associated model score.
    Args:
        points (np.array): (num_points, num_dims)-sized array of points evaluated so far
        scores (np.array): (num_points)-sized array of scores from evaluated points
    """
    X = np.hstack((points, scores[:,np.newaxis]))
    np.savetxt("scores.csv", X, fmt='%.6e', delimiter=',')

def load_checkpoint(model, scoresfile):
    """
    Load the best model, evaluated points, and evaluated scores from a Gaussian Process Bandits checkpoint file
    Args:
        model: a data model
        scoresfile (String): path to a scores file to load from
    Returns:
        model: a data model with the best hyperparameters set
        points (np.array): (num_points, num_dims)-sized array of points evaluated so far
        scores (np.array): (num_points)-sized array of scores from evaluated points
    """
    # load data from scores file
    X = np.loadtxt(scoresfile, delimiter=',')

    # separate into points and scores
    scores = X[:,-1]
    points = X[:,:-1]

    # set best hyperparameters based on best scores
    ind = np.argmin(scores)
    best_overall_point = points[ind]
    model.decode(best_overall_point)

    return model, points, scores
