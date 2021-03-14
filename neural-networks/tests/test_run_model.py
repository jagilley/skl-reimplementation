import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data.my_dataset import MyDataset
from torch.utils.data import DataLoader
import torch.optim as optim

def test_train_function():
	"""
	Test the ability of the 'train' function to train a simple network (1 epoch).
	"""

	from src.run_model import _train

	trainX = np.array([[10,0],[10,0.1],[10,-0.1],[9.9,0],[10.1,0],
		[-10,0],[-10,0.1],[-10,-0.1],[-9.9,1],[-10.1,0]])
	trainY = np.array([0,0,0,0,0,1,1,1,1,1])

	train_dataset = MyDataset(trainX, trainY)
	train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

	model = Basic_Model(weight_init=0.001)

	optimizer = optim.SGD(model.parameters(), lr=1e-1)

	_, _est_loss, _est_acc = _train(model,train_loader,optimizer)
	_est_values = np.array([_est_loss, _est_acc])

	_true_values = np.array([0.70840007, 50.0]) 

	assert np.allclose(_true_values, _est_values)


def test_test_function():
	"""
	Test the ability of the 'test' function to compute the loss and accuracy.
	"""
	from src.run_model import _test

	testX = np.array([[1,1],[-1,1],[-1,-1],[1,-1]])
	testY = np.array([1,0,0,0])

	test_dataset = MyDataset(testX, testY)
	test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

	model = Basic_Model(weight_init=1)

	_est_loss, _est_acc = _test(model, test_loader)
	_est_values = np.array([_est_loss, _est_acc])

	_true_values = np.array([0.69314718, 25.0]) 

	assert np.allclose(_true_values, _est_values)


def test_run_model():
	"""
	Test the ability of the 'run_model' function to train a simple network (5 epochs).
	"""

	from src.run_model import run_model

	trainX = np.array([[0,5],[0,4.9],[0,5.1],[0.1,5],[-0.1,5],
		[0,-5],[0,-4.9],[0,-5.1],[0.1,-5],[-0.1,-5]])
	trainY = np.array([0,0,0,0,0,1,1,1,1,1])

	validX = np.array([[-1,4.5],[-3,3],[-2,6],[1,-0.5]])
	validY = [0,0,0,1]

	train_dataset = MyDataset(trainX, trainY)
	valid_dataset = MyDataset(validX, validY)

	model = Basic_Model(weight_init=0.5)

	_, _est_loss, _est_acc = run_model(model,running_mode='train', train_set=train_dataset, 
		valid_set = valid_dataset, batch_size=1, learning_rate=1e-3, 
		n_epochs=5, shuffle=False)

	_est_loss_train = np.mean(_est_loss['train'])
	_est_loss_valid = np.mean(_est_loss['valid'])

	_est_acc_train = np.mean(_est_acc['train'])
	_est_acc_valid = np.mean(_est_acc['valid'])

	_est_values = np.array([_est_loss_train, _est_loss_valid, _est_acc_train, _est_acc_valid])

	_true_values = np.array([0.63108519, 0.63902633, 60., 50.])

	assert np.allclose(_true_values, _est_values)


class Basic_Model(nn.Module):
	"""
	A very simple model for testing purposes.
	"""
	def __init__(self, weight_init=0.001):
		super(Basic_Model, self).__init__()

		self.hidden = nn.Linear(2,4)
		self.out = nn.Linear(4,2)

		self.hidden.weight.data.fill_(weight_init)
		self.hidden.bias.data.fill_(0.0)

		self.out.weight.data.fill_(weight_init)
		self.out.bias.data.fill_(0.0)

	def forward(self, input):

		hidden_out = F.relu(self.hidden(input))
		output = self.out(hidden_out)

		return output


