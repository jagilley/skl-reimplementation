import torch.nn as nn
import matplotlib.pyplot as plt
from data.load_data import load_mnist_data
from src.models import Digit_Classifier
from data.my_dataset import MyDataset
import time
import numpy as np
import matplotlib
from src.run_model import run_model

times = []

for examples_no in [50, 100, 150, 200]:
    start = time.time()

    train_features, test_features, train_targets, test_targets = load_mnist_data(10, examples_per_class=examples_no, fraction=0.5)

    train_dataset = MyDataset(train_features, train_targets.astype(int))
    valid_dataset = MyDataset(test_features, test_targets.astype(int))

    #print(valid_dataset.x, valid_dataset.y)

    model = Digit_Classifier()

    _, _est_loss, _est_acc = run_model(model, running_mode='train', train_set=train_dataset, 
        valid_set = valid_dataset, batch_size=1, learning_rate=1e-3, 
        n_epochs=100, shuffle=True)
    
    end = time.time()
    _est_acc_valid = np.mean(_est_acc['valid'])

    times.append(_est_acc_valid)

fig, ax = plt.subplots()
ax.plot([50, 100, 150, 200], times)

ax.set(xlabel='Number of training examples', ylabel='Accuracy',
       title='Accuracy vs. training examples')
ax.grid()

fig.savefig("experiments/outputs/2.png")
plt.show()