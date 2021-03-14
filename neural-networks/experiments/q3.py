import torch.nn as nn
import matplotlib.pyplot as plt
from data.load_data import load_mnist_data
from src.models import Digit_Classifier, Dog_Classifier_FC, Dog_Classifier_Conv
from data.my_dataset import MyDataset
import time
from data.dogs import DogsDataset
import numpy as np
import matplotlib
from src.run_model import run_model

dogz = DogsDataset("data/DogSet")

trainX, trainY = dogz.get_train_examples()
testX, testY = dogz.get_test_examples()
validX, validY = dogz.get_validation_examples()

times = []

start = time.time()

train_dataset = MyDataset(trainX, trainY.astype(int))
valid_dataset = MyDataset(testX, testY.astype(int))

kernel_size = [(5,5),(5,5)]
stride = [(1,1),(1,1)]

model = Dog_Classifier_Conv(kernel_size, stride)

_, _est_loss, _est_acc = run_model(model, running_mode='train', train_set=train_dataset, 
    valid_set = valid_dataset, batch_size=1, learning_rate=1e-4, 
    n_epochs=60, shuffle=True)

end = time.time()
_est_acc_valid = np.mean(_est_acc['valid'])

times.append(_est_acc_valid)

fig, ax = plt.subplots()
ax.plot(list(range(len(_est_acc['valid']))), _est_acc["train"])
ax.plot(list(range(len(_est_loss['valid']))), _est_acc["valid"])

ax.set(xlabel='Epoch', ylabel='Accuracy',
       title='Accuracy vs. epoch')
ax.grid()

fig.savefig("experiments/outputs/6.png")
plt.show()

fig, ax = plt.subplots()
ax.plot(list(range(len(_est_acc['valid']))), _est_loss["train"])
ax.plot(list(range(len(_est_loss['valid']))), _est_loss["valid"])

ax.set(xlabel='Epoch', ylabel='Accuracy',
       title='Loss vs. epoch')
ax.grid()

fig.savefig("experiments/outputs/7.png")
plt.show()

print(_est_loss["valid"])
print(_est_loss["train"])
print([r.item() for r in _est_acc["valid"]])
print([r.item() for r in _est_acc["train"]])