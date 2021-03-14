import torch.nn as nn
import matplotlib.pyplot as plt
from data.load_data import load_synth_data
from src.models2 import Digit_Classifier, Dog_Classifier_FC, Dog_Classifier_Conv, Synth_Classifier
from data.my_dataset import MyDataset
import time
from data.dogs import DogsDataset
import numpy as np
import matplotlib
import random
from src.run_model import run_model

trainX, trainY = load_synth_data("synth_data")
"""
#print(trainX[0])
#print(trainY[0])

for i, image in enumerate(trainX):
    no = random.randint(0, len(trainX))
    guy = trainX[no]
    plt.imshow(guy, cmap='gray')
    plt.show()
    print("trainY[i] is", trainY[no])
    
    if i == 7:
        break

exit()
"""
times = []

start = time.time()

train_dataset = MyDataset(np.expand_dims(trainX, axis=3), trainY.astype(int))
#valid_dataset = MyDataset(testX, testY.astype(int))

kernel_size = [(5,5),(3,3),(3,3)]
stride = [(1,1),(1,1),(1,1)]

model = Synth_Classifier(kernel_size, stride)

mdl, _est_loss, _est_acc = run_model(model, running_mode='train', train_set=train_dataset, 
    valid_set=train_dataset, batch_size=50, learning_rate=1e-4, 
    n_epochs=60, shuffle=False)

w0 = mdl.c1.weight[0].reshape(-1)
w1 = mdl.c1.weight[1].reshape(-1)

print(w0, w1)

fig, ax = plt.subplots()
ax.plot(list(range(len(w0))), w0.detach().numpy())
ax.plot(list(range(len(w1))), w1.detach().numpy())

ax.set(xlabel='Index', ylabel='Weight',
       title='Kernel Weights')
ax.grid()

fig.savefig("experiments/outputs/91.png")
plt.show()
"""

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
"""