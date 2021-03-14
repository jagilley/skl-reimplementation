from your_code import GradientDescent, load_data
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from your_code import HingeLoss, ZeroOneLoss

train_features, test_features, train_targets, test_targets = load_data('synthetic', fraction=1)

train_features = np.array([[3, 4, 5, 6]]).T

print(train_targets)
train_targets = train_targets[2:]
print(train_targets)

liszt = [0.5, -0.5, -1.5, -2.5, -3.5, -4.5, -5.5]

out = []
for i in [0.5, -0.5, -1.5, -2.5, -3.5, -4.5, -5.5]:
    onez = np.ones((4,1))
    #onez[-1] = i
    tf2 = np.hstack((train_features, onez))
    print(tf2)
    testt = ZeroOneLoss()
    out.append(testt.forward(tf2, np.array([1, i]), train_targets))

fig, ax = plt.subplots()
ax.plot(liszt, out)

ax.set(xlabel='Bias', ylabel='Loss', title='Question 2')
ax.grid()

fig.savefig("outputs/q2-3.png")
plt.show()