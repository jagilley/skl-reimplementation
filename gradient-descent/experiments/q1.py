from your_code import GradientDescent, load_data
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

train_features, test_features, train_targets, test_targets = load_data('mnist-binary', fraction=0.5)

accz = []

learner = GradientDescent('hinge')
losses = learner.fit(train_features, train_targets, extra=2, tf=test_features)

print('accs are', losses)
#predictions = learner.predict(test_features)

fig, ax = plt.subplots()
ax.plot(list(range(len(losses))), losses)

ax.set(xlabel='Epoch', ylabel='Loss', title='Question 1')
ax.grid()

fig.savefig("outputs/q1-7.png")
plt.show()