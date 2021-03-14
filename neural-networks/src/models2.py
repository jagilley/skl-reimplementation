import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Digit_Classifier(nn.Module):
    """
    This is the class that creates a neural network for classifying handwritten digits
    from the MNIST dataset.
	
	Network architecture:
	- Input layer
	- First hidden layer: fully connected layer of size 128 nodes
	- Second hidden layer: fully connected layer of size 64 nodes
	- Output layer: a linear layer with one node per class (in this case 10)

	Activation function: ReLU for both hidden layers

    """
    def __init__(self):
        super(Digit_Classifier, self).__init__()
        self.l1 = nn.Linear(28 * 28, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 10)

    def forward(self, input):
        #input = input.unsqueeze(0)
        x = self.l1(input)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        #print(x)
        #output = F.log_softmax(x, dim=1)
        return x

class Dog_Classifier_FC(nn.Module):
    """
    This is the class that creates a fully connected neural network for classifying dog breeds
    from the DogSet dataset.
    
    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """

    def __init__(self):
        super(Dog_Classifier_FC, self).__init__()
        self.l1 = nn.Linear(64 * 64 * 3, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 10)

    def forward(self, input):
        #input = input.unsqueeze(0)
        b = input.reshape((input.shape[0], 64*64*3))
        x = self.l1(b)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        #print(x)
        #output = F.log_softmax(x, dim=1)
        return x


class Dog_Classifier_Conv(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying dog breeds
    from the DogSet dataset.
    
    Network architecture:
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers
    
    There should be a maxpool after each convolution. 
    
    The sequence of operations looks like this:
    
    	1. Apply convolutional layer with stride and kernel size specified
		- note: uses hard-coded in_channels and out_channels
		- read the problems to figure out what these should be!
	2. Apply the activation function (ReLU)
	3. Apply 2D max pooling with a kernel size of 2

    Inputs: 
    kernel_size: list of length 2 containing kernel sizes for the two convolutional layers
                 e.g., kernel_size = [(3,3), (3,3)]
    stride: list of length 2 containing strides for the two convolutional layers
            e.g., stride = [(1,1), (1,1)]

    """

    def __init__(self, kernel_size, stride):
        super(Dog_Classifier_Conv, self).__init__()
        #self.l1 = nn.Linear(64 * 64 * 3, 128)
        self.c1 = nn.Conv2d(3, 16, kernel_size=kernel_size[0], stride=stride[0])
        self.c2 = nn.Conv2d(16, 32, kernel_size=kernel_size[1], stride=stride[1])
        self.l3 = nn.Linear(5408, 10)

    def forward(self, input):
        b = input.permute(0, 3, 1, 2)
        x = self.c1(b)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.c2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        asdf = x.view(x.size(0), -1)
        x = self.l3(asdf)
        return x


class Synth_Classifier(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying 
    synthesized images.
    
    Network architecture:
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 2)

    Activation function: ReLU for both hidden layers
    
    There should be a maxpool after each convolution. 
    
    The sequence of operations looks like this:
    
    	1. Apply convolutional layer with stride and kernel size specified
		- note: uses hard-coded in_channels and out_channels
		- read the problems to figure out what these should be!
	2. Apply the activation function (ReLU)
	3. Apply 2D max pooling with a kernel size of 2

    Inputs: 
    kernel_size: list of length 3 containing kernel sizes for the three convolutional layers
                 e.g., kernel_size = [(5,5), (3,3),(3,3)]
    stride: list of length 3 containing strides for the three convolutional layers
            e.g., stride = [(1,1), (1,1),(1,1)]

    """

    def __init__(self, kernel_size, stride):
        super(Synth_Classifier, self).__init__()
        self.c1 = nn.Conv2d(1, 2, kernel_size=kernel_size[0], stride=stride[0])
        self.c2 = nn.Conv2d(2, 4, kernel_size=kernel_size[1], stride=stride[1])
        self.c3 = nn.Conv2d(4, 8, kernel_size=kernel_size[2], stride=stride[2])
        self.l3 = nn.Linear(8, 2)
        
    def forward(self, input):
        #print(input.shape)
        b = input.permute(0, 3, 2, 1)
        x = self.c1(b)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.c2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.c3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        asdf = x.view(x.size(0), -1)
        x = self.l3(asdf)
        return x