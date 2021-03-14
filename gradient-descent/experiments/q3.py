from your_code import GradientDescent, load_data
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from your_code import HingeLoss, ZeroOneLoss
from your_code import confusion_matrix
from your_code import MultiClassGradientDescent, accuracy

train_features, test_features, train_targets, test_targets = load_data('mnist-multiclass', fraction=0.75)

learner = MultiClassGradientDescent(loss='squared', regularization=None, learning_rate=0.01, reg_param=0.05)
learner.fit(train_features, train_targets, batch_size=None, max_iter=1000)
predictions = learner.predict(test_features)

print(confusion_matrix(test_targets, predictions))