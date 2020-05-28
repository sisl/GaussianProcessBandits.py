import numpy as np
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms
import torch.utils
import torch.optim

class NeuralNet:

    """
    Class defining/Encapsulation of the fully connected neural network for the Black Box Optimization.

    Hyperparemetrs (in order):

        -number of hidden layers
        -number of hidden units
        -activation functions: ReLU or TanH
        -dropout probability (probability of an element being zeroed out)

    """
    def __init__(self, input_dim, num_classes):
        """
        Initializes model parameters.
        Args:
        """
        #Initial model hyperparemetrs

        self.num_dims = 4
        self.max_hid_layers = 5
        self.max_hid_units = 128
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

        self.decode(point)

    def decode(self, point):
        """
        Decodes the parameter point (2,) from black-box optimization scale [0,1) to
        ridge_reg class scale [1e-15, 1e3).

        Args:
            point (np.array): (1,)-sized numpy array with hyperparameters in the
            black-box optimization scale
        """

        self.num_hid_layers = int(point[0] * self.max_hid_layers)
        self.num_hid_units = int(point[1] * self.max_hid_units)
        self.activation = nn.ReLU() if point[2] // 0.5 == 0.0 else nn.Tanh()
        self.drop_prob = point[3]

        #print(self.input_dim," ",self.num_classes," ", self.num_hid_layers, " ",
        #                                self.num_hid_units," ", self.activation," ", self.drop_prob)

        self.current_point = point ## Check when copy and when passed by refernce.???

        self.model_ = PyTorchNetwork(self.input_dim, self.num_classes, self.num_hid_layers,
                                        self.num_hid_units, self.activation, self.drop_prob)

    def encode(self):
        """
        Encodes the parameter point from ridge_reg class scale [1e-15, 1e3) to
        black-box optimization scale [0,1).

        Returns:
            point (np.array): (1,)-sized numpy array with hyperparameters in the
            black-box optimization scale
        """
        self.print()
        return self.current_point

    def print(self):
        print("Model. Num hid layers: {}, Num hid units: {}, Activation: {}, Drop Prob: {}".format(self.num_hid_layers, self.num_hid_units, self.activation, self.drop_prob))

    def train_test_cv(self, data):
        """
        Fits and evaluates the model to the dataset provided by the
        trainloader/testloader.
        """

        train_loader = data.train_loader
        test_loader = data.test_loader

        lr = 0.001
        momentum = 0.9
        max_epochs = 2

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model_.parameters(), lr=lr, momentum=momentum) #Modify

        self.model_.train()

        #Training

        print("Training...")
        for epoch in range(max_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model_(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            # print('Epoch %d, Training loss: %.3f' %
            #       (epoch + 1, running_loss))
            running_loss = 0.0

        #Test

        print("Testing...")

        self.model_.eval()

        with torch.no_grad():
            test_loss = 0.0
            for i, data in enumerate(test_loader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # forward + backward + optimize
                outputs = self.model_(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

            print("Test Loss: {}".format(test_loss))
            ### Check for convergence.

        return test_loss


    def predict(self,X):
        """
        Predicts the target_variable based on the provided data matrix X and
        learned parameters theta.

        I don't think it is needed.

        Args:
            X (np.array): (n_samples, height, width)-numpy array features with data

        Returns:
            y (np.array): (n_samples,)-numpy array with predicted target variable
        """
        y_pred = self.model_(X)

        return y_pred



class PyTorchNetwork(nn.Module):
    """
    Doc
    """
    def __init__(self, input_dim, output_dim, num_hid_layers, num_hid_units, activation, drop_prob):
        """
        Doc
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
        Doc
        """
        y = self.model(x)
        return y

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)
