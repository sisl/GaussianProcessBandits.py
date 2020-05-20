class linear_l2:
    
    def self.__init__():
        # set initial model hyperparameters
        self.num_dims = 1
        self.lambda = 0.5
        
        
        
        
        
    def self.decode(point):
        # set hyperparameters
        self.lambda = np.exp(10*point-5)
        
        pass
    
    def self.encode():
        return point
    
    def self.train_and_test_cv(data):
        
        for k in fold:
            
            # solve for theta using closed form with lambda
            # theta inv(xtx + lambdaI)xtY
            # measure error on remaining fold
            pass
            
        
         # average error 
         return error