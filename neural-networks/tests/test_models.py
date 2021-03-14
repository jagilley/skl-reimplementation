import numpy as np 
import torch

def test_digit_classifier():
	"""
	This function checks the architecture and forward method of the digit classifier.
	"""

	from src.models import Digit_Classifier

	model = Digit_Classifier()

	model_params = model.state_dict()

	model_weight_shapes = []
	model_bias_shapes = []

	for i in model_params.keys():
		if 'weight' in i: 
			weight_shape = model_params[i].detach().numpy().shape
			model_weight_shapes.append(weight_shape)

			getattr(model,i.split('.')[0]).weight.data.fill_(0.01)

		elif 'bias' in i:
			bias_shape = model_params[i].detach().numpy().shape
			model_bias_shapes.append(bias_shape)

			getattr(model,i.split('.')[0]).bias.data.fill_(0)


	true_weight_shapes = [(128, 784), (64, 128), (10, 64)]
	true_bias_shapes = [(128,), (64,), (10,)]

	input = torch.Tensor(0.01 * np.ones((1,784)))   
	
	_est = model.forward(input)
	_est_mean = np.mean(_est.detach().numpy())

	_true_mean = 0.064225264

	assert np.all(model_weight_shapes == true_weight_shapes)
	assert np.all(model_bias_shapes == true_bias_shapes)
	assert np.allclose(_est_mean, _true_mean)


def test_dog_classifier_fc():
	"""
	This function checks the architecture and forward method of the fully connected 
	dog classifier.
	"""

	from src.models import Dog_Classifier_FC

	model = Dog_Classifier_FC()

	model_params = model.state_dict()

	model_weight_shapes = []
	model_bias_shapes = []

	for i in model_params.keys():
		if 'weight' in i: 
			weight_shape = model_params[i].detach().numpy().shape
			model_weight_shapes.append(weight_shape)

			getattr(model,i.split('.')[0]).weight.data.fill_(0.01)

		elif 'bias' in i:
			bias_shape = model_params[i].detach().numpy().shape
			model_bias_shapes.append(bias_shape)

			getattr(model,i.split('.')[0]).bias.data.fill_(0)


	true_weight_shapes = [(128, 12288), (64, 128), (10, 64)]
	true_bias_shapes = [(128,), (64,), (10,)]

	input = torch.Tensor(0.1 * np.ones((1,12288))) 

	_est = model.forward(input)
	_est_mean = np.mean(_est.detach().numpy())  

	_true_mean = 10.066246

	assert np.all(model_weight_shapes == true_weight_shapes)
	assert np.all(model_bias_shapes == true_bias_shapes)
	assert np.allclose(_est_mean, _true_mean)


def test_dog_classifier_conv():
	"""
	This function checks the architecture and forward method of the convolutional 
	dog classifier.
	"""

	from src.models import Dog_Classifier_Conv

	kernel_size = [(5,5),(5,5)]
	stride = [(1,1),(1,1)]
	model = Dog_Classifier_Conv(kernel_size, stride)

	model_params = model.state_dict()

	model_weight_shapes = []
	model_bias_shapes = []

	for i in model_params.keys():
		if 'weight' in i: 
			weight_shape = model_params[i].detach().numpy().shape
			model_weight_shapes.append(weight_shape)

			if '0' in i:
				getattr(model,i.split('.')[0])[0].weight.data.fill_(0.01)
			else:
				getattr(model,i.split('.')[0]).weight.data.fill_(0.01)

		elif 'bias' in i:
			bias_shape = model_params[i].detach().numpy().shape
			model_bias_shapes.append(bias_shape)

			if '0' in i:
				getattr(model,i.split('.')[0])[0].bias.data.fill_(0)
			else:
				getattr(model,i.split('.')[0]).bias.data.fill_(0)


	true_weight_shapes = [(16, 3, 5, 5), (32, 16, 5, 5), (10, 5408)]
	true_bias_shapes = [(16,), (32,), (10,)]

	input = torch.Tensor(0.1 * np.ones((1,64,64,3)))

	_est = model.forward(input)
	_est_mean = np.mean(_est.detach().numpy())  

	_true_mean = 16.224094

	assert np.all(model_weight_shapes == true_weight_shapes)
	assert np.all(model_bias_shapes == true_bias_shapes)
	assert np.allclose(_est_mean, _true_mean)


def test_synth_classifier():
	"""
	This function checks the architecture and forward method of the convolutional 
	synth classifier.
	"""

	from src.models import Synth_Classifier

	kernel_size = [(3,3),(3,3),(3,3)]
	stride = [(1,1),(1,1),(1,1)]

	model = Synth_Classifier(kernel_size, stride)

	model_params = model.state_dict()

	model_weight_shapes = []
	model_bias_shapes = []

	for i in model_params.keys():
		if 'weight' in i: 
			weight_shape = model_params[i].detach().numpy().shape
			model_weight_shapes.append(weight_shape)

			if '0' in i:
				getattr(model,i.split('.')[0])[0].weight.data.fill_(0.1)
			else:
				getattr(model,i.split('.')[0]).weight.data.fill_(0.1)

		elif 'bias' in i:
			bias_shape = model_params[i].detach().numpy().shape
			model_bias_shapes.append(bias_shape)

			if '0' in i:
				getattr(model,i.split('.')[0])[0].bias.data.fill_(0)
			else:
				getattr(model,i.split('.')[0]).bias.data.fill_(0)


	true_weight_shapes = [(2, 1, 3, 3), (4, 2, 3, 3), (8, 4, 3, 3), (2, 8)]
	true_bias_shapes = [(2,), (4,), (8,), (2,)]

	input = torch.Tensor(np.ones((1,28,28,1)))

	_est = model.forward(input)
	_est_mean = np.mean(_est.detach().numpy())  

	_true_mean = 4.6656017

	assert np.all(model_weight_shapes == true_weight_shapes)
	assert np.all(model_bias_shapes == true_bias_shapes)
	assert np.allclose(_est_mean, _true_mean)





