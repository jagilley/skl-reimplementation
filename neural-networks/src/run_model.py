import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import numpy as np

def run_model(model,running_mode='train', train_set=None, valid_set=None, test_set=None, 
	batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True):
	"""
	This function either trains or evaluates a model. 

	training mode: the model is trained and evaluated on a validation set, if provided. 
				   If no validation set is provided, the training is performed for a fixed 
				   number of epochs. 
				   Otherwise, the model should be evaluted on the validation set 
				   at the end of each epoch and the training should be stopped based on one
				   of these two conditions (whichever happens first): 
				   1. The validation loss stops improving. 
				   2. The maximum number of epochs is reached.

    testing mode: the trained model is evaluated on the testing set

    Inputs: 

    model: the neural network to be trained or evaluated
    running_mode: string, 'train' or 'test'
    train_set: the training dataset object generated using the class MyDataset 
    valid_set: the validation dataset object generated using the class MyDataset
    test_set: the testing dataset object generated using the class MyDataset
    batch_size: number of training samples fed to the model at each training step
	learning_rate: determines the step size in moving towards a local minimum
    n_epochs: maximum number of epoch for training the model 
    stop_thr: if the validation loss from one epoch to the next is less than this
              value, stop training
    shuffle: determines if the shuffle property of the DataLoader is on/off

    Outputs when running_mode == 'train':

    model: the trained model 
    loss: dictionary with keys 'train' and 'valid'
    	  The value of each key is a list of loss values. Each loss value is the average
    	  of training/validation loss over one epoch.
    	  If the validation set is not provided just return an empty list.
    acc: dictionary with keys 'train' and 'valid'
    	 The value of each key is a list of accuracies (percentage of correctly classified
    	 samples in the dataset). Each accuracy value is the average of training/validation 
    	 accuracies over one epoch. 
    	 If the validation set is not provided just return an empty list.

    Outputs when running_mode == 'test':

    loss: the average loss value over the testing set. 
    accuracy: percentage of correctly classified samples in the testing set. 
	
	Summary of the operations this function should perform:
	1. Use the DataLoader class to generate trainin, validation, or test data loaders
	2. In the training mode:
	   - define an optimizer (we use SGD in this homework)
	   - call the train function (see below) for a number of epochs untill a stopping
	     criterion is met
	   - call the test function (see below) with the validation data loader at each epoch 
	     if the validation set is provided

    3. In the testing mode:
       - call the test function (see below) with the test data loader and return the results

	"""

	#train_dataset = MyDataset(train_set, valid_set)
	loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
	test_loadr = DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle)
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)

	if running_mode == "train":
		old_loss = -314
		train_loss_list = []
		train_acc_list = []
		test_loss_list = []
		test_acc_list = []
		for i in range(n_epochs):
			model, train_loss, train_acc = _train(model, loader, optimizer)
			train_loss_list.append(train_loss)
			train_acc_list.append(train_acc)
			
			test_loss, test_acc = _test(model, test_loadr)
			test_loss_list.append(test_loss)
			test_acc_list.append(test_acc)

			print(f"Epoch {i}: train_loss {train_loss}, train_acc {train_acc}, test_loss {test_loss}, test_acc {test_acc}")

			if old_loss == test_loss:
				break
				#pass
			old_loss = test_loss
		return model, {"train": train_loss_list, "valid": test_loss_list}, {"train": train_acc_list, "valid": test_acc_list}
	elif running_mode == "test":
		pass
	else:
		raise AssertionError("Invalid running_mode")


def _train(model,data_loader,optimizer,device=torch.device('cpu')):
	"""
	This function implements ONE EPOCH of training a neural network on a given dataset.
	Example: training the Digit_Classifier on the MNIST dataset

	Inputs:
	model: the neural network to be trained
	data_loader: for loading the netowrk input and targets from the training dataset
	optimizer: the optimiztion method, e.g., SGD 
	device: we run everything on CPU in this homework

	Outputs:
	model: the trained model
	train_loss: average loss value on the entire training dataset
	train_accuracy: average accuracy on the entire training dataset
	"""

	# use torch.nn.functional.cross_entropy instead
	#lo55 = CrossEntropyLoss()
	lo55 = F.cross_entropy
	#output = loss(input, target)
	#for batch in data_loader:
	#	fwd = model.forward(batch)
	running_loss = 0.0
	running_acc = 0
	for i, data in enumerate(data_loader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data
		inputs = inputs.float()

		# zero the parameter gradients
		optimizer.zero_grad()

		outputs = model(inputs)
		loss = lo55(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		
		#inp_clean = inputs.detach().numpy()[0]#.item()
		#out_clean = outputs.detach().numpy()[0]#.item()
		#print(inp_clean, out_clean)
		#my_acc = ((inp_clean.eq(out_clean.long())).sum()*100).item()
		#print(my_acc.item())
		#my_acc = np.sum(inp_clean == out_clean)/len(out_clean)*100

		ps = torch.exp(outputs)
		equality = (labels.data == ps.max(dim=1)[1])
		my_acc = equality.type(torch.FloatTensor).mean()

		running_acc += my_acc*100

	#print(running_loss, running_acc)

	return model, running_loss/len(data_loader), running_acc/len(data_loader)


def _test(model, data_loader, device=torch.device('cpu')):
	"""
	This function evaluates a trained neural network on a validation set
	or a testing set. 

	Inputs:
	model: trained neural network
	data_loader: for loading the netowrk input and targets from the validation or testing dataset
	device: we run everything on CPU in this homework

	Output:
	test_loss: average loss value on the entire validation or testing dataset 
	test_accuracy: percentage of correctly classified samples in the validation or testing dataset
	"""

	lo55 = F.cross_entropy
	#output = loss(input, target)
	#for batch in data_loader:
	#	fwd = model.forward(batch)
	running_loss = 0.0
	running_acc = 0
	for i, data in enumerate(data_loader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data
		inputs = inputs.float()

		# zero the parameter gradients
		#optimizer.zero_grad()

		outputs = model(inputs)
		loss = lo55(outputs, labels)
		#loss.backward()
		#optimizer.step()

		running_loss += loss.item()

		#labels_T = labels.view(len(labels), 1)
		#inputs_col = inputs[:, :1].view(len(inputs))
		#my_acc = (inputs_col.eq(labels.long())).sum()
		#my_acc = my_acc.item()/len(inputs)*100

		ps = torch.exp(outputs)
		equality = (labels.data == ps.max(dim=1)[1])
		my_acc = equality.type(torch.FloatTensor).mean()

		running_acc += my_acc*100

	return running_loss/len(data_loader), running_acc/len(data_loader)