def gpbandits(model, data, , max_iters, kernel='se', cl=2.0, v=0.0):
    """
		Chooses ...
		
        Args:
			samples (torch.tensor): (nsamps,*) samples
		Returns:
			log_probs (torch.tensor): (nsamps,) log probabilties
		"""
    
    num_dims = model.num_dims # number of hyperparater dimensions
    points = []
    scores = []
    
    # initial model evaluation
    

    
    # loop
    for i in range(max_iters):
    
    
        # sample m random points from [0,1]^num_dims
    
        # finding GP posterior
        
        # choose new point with best expected improvement
        
        # evaluating model with new point
        
        model.decode(point) # set hyperparameters
        
        point = model.encode() # return re-encoded point
        
        score = model.train_test_cv(data) # return score to be minimized
        