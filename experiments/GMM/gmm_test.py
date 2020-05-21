import numpy as np
from sklearn import datasets

import sys
sys.path.append("../../GaussianProcessBandits")
from gpbandits import *
from models.gmm import *

# Make dataset with 10 centers
blobs, labels = datasets.make_blobs(n_samples=2500, n_features=4, centers=10, random_state=8)

# initialize model
model = GMM()

# run bandits
best_model = gpbandits(model, blobs, iters=10, kernel='se', cl=0.1, v=0.1)
