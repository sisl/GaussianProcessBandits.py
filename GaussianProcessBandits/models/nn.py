import numpy as np
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms
import torch.utils
import torch.optim

class NeuralNet:

    """
    Class defining/Encapsulation of the fully connected neural network for the classification task used in the black box optimization.

    Optimized Hyperparemetrs:

        -number of hidden layers
        -number of hidden units per layer
        -activation functions: ReLU or TanH
        -dropout probability (probability of an element being zeroed out)

    """
    def __init__(self, input_dim, num_classes):
        """
        Initializes model parameters.
        Args:
            input_dim (int) : input dimension
            num_classes (int) : number of classes
        """

        #Initial model hyperparemetrs
        self.num_dims = 4
        self.max_hid_layers = 2
        self.max_hid_units = 32
        self.input_dim = input_dim
        self.num_classes = num_classes


        #Initializes the point.
        point = np.zeros((self.num_dims,))

        #Number of hidden layers:
        point[0] = 0.5
        #Number of hidden units:
        point[1] = 0.5
        #Activation function:
        point[2] = 0.25
        #Dropout probability
        point[3] = 0.1

        if torch.cuda.is_available():
            print('Using GPU.')
        else:
            print('Warning: using CPU.')

        self.decode(point)

    def decode(self, point):
        """
        Decodes the parameter point from black-box optimization scale [0,1)
        to neural network PyTorch parameters.

        Args:
            point (np.array): (4,)-sized numpy array with hyperparameters in the
            black-box optimization scale
        """

        self.num_hid_layers = int(point[0] * self.max_hid_layers) + 1
        self.num_hid_units = int(point[1] * self.max_hid_units) + 1
        self.activation = nn.ReLU() if point[2] // 0.5 == 0.0 else nn.Tanh()
        self.drop_prob = point[3]

        self.print()

        self.current_point = point ## Check when copy and when passed by refernce.???

        self.model_ = PyTorchNetwork(self.input_dim, self.num_classes, self.num_hid_layers,
                                        self.num_hid_units, self.activation, self.drop_prob)
        if torch.cuda.is_available():
            self.model_ = self.model_.cuda()


    def encode(self):
        """
        Encodes the parameter point from neural network PyTorch parameters to
        black-box optimization scale [0,1).

        Returns:
            point (np.array): (4,)-sized numpy array with hyperparameters in the
            black-box optimization scale
        """
        return self.current_point

    def print(self):
        """
        Prints the model architecture.
        """

        print("       Model. Num hid layers: {}, Num hid units: {}, Activation: {}, Drop Prob: {}".format(
        self.num_hid_layers, self.num_hid_units, self.activation, self.drop_prob))

    def train_test_cv(self, data):
        """
        Return model score (to be minimized) with data data using set hyperparameters

        Args:
            data (Dataset class): Dataset object with Pytorch's train loader and test loader
        Returns:
            score (float): the score of the fit model (to be minimized)
        """

        train_loader = data.train_loader
        test_loader = data.test_loader

        lr = 0.001
        #momentum = 0.5
        max_epochs = 20
        #prev_loss = 0

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model_.parameters(), lr=lr) #Modify

        model = self.model_

        model.train()

        #Training
        for epoch in range(max_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                inputs, labels = inputs.cuda(), labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()


            # if np.abs((running_loss - prev_loss)/running_loss) < 0.01:
            #     #Converged
            #     break

            #print("training loss: ", running_loss)

            #prev_loss = running_loss
            running_loss = 0.0

        #Test
        model.eval()

        with torch.no_grad():
            test_loss = 0.0
            for i, data in enumerate(test_loader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                inputs, labels = inputs.cuda(), labels.cuda()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

            #print("Test Loss: {}".format(test_loss))

        self.model_ = model

        return test_loss


    def predict(self,X):
        """
        Predicts the target_variable based on the provided data matrix X and
        learned parameters theta.

        I don't think it is needed/FIX IT.

        Args:
            X (np.array): (n_samples, n_features)-numpy array features with data

        Returns:
            y (np.array): (n_samples,)-numpy array with class scores
        """

        #### APPLY NORMALIZATION!!!!


        y_pred = self.model_(X.cuda())

        return y_pred



class PyTorchNetwork(nn.Module):
    """
    Class defining a fully connected network for the classification taks in PyTorch.

    """

    def __init__(self, input_dim, output_dim, num_hid_layers, num_hid_units, activation, drop_prob):
        """
        Initializes the network.

        Args:
            input_dim (int) : dimension of the input data
            output_dim (int) : dimension of the output data/number of classes
            num_hid_layers (int) : number of hidden layers
            num_hid_units (int) : number of hidden units
            activation (PyTorch nn.Module) : nonlinearity applied between layers
            drop_prob (float) : probability of element being zeroed out in Dropout module
        """

        super(PyTorchNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hid_layers
        self.num_hid_units = num_hid_units
        self.activation = activation
        self.drop_prob = drop_prob


        modules = []

        # Input layer/hidden layers
        modules.append(nn.Linear(self.input_dim, self.num_hid_units))
        modules.append(self.activation)
        modules.append(nn.Dropout(p = self.drop_prob))

        for _ in range(self.num_hidden_layers):
            modules.append(nn.Linear(self.num_hid_units, self.num_hid_units))
            modules.append(self.activation)
            modules.append(nn.Dropout(p = self.drop_prob))

        ## Output Layer
        modules.append(nn.Linear(self.num_hid_units, self.output_dim))
        modules.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        """
        Foward pass.

        Returns:
            y (PyTorch tensor) : output tensor
        """
        y = self.model(x)
        return y

class ReshapeTransform:
        """
        Class reshaping the vector which can be applied in datapreprocessing pipeline.
        See jupyter notebook for example.
        """

        def __init__(self, new_size):
            self.new_size = new_size

        def __call__(self, img):
            return torch.reshape(img, self.new_size)
