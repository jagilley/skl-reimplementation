import gym
from code import MultiArmedBandit, QLearning
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

env = gym.make('FrozenLake-v0')

def getavg(no=1, algo=MultiArmedBandit, epsilon=0.2, adap=False):
    lists = []
    out = pd.DataFrame()
    for i in range((no+1)):
        agent = algo(epsilon=epsilon, adaptive=adap)
        action_values, rewards = agent.fit(env, steps=100000)
        lists.append(rewards)
    df = pd.DataFrame(lists, columns=list(range(100)))
    return list(df.mean())

def lrl(arr):
    return list(range(len(arr)))

print("group1")

#guy1 = getavg(no=1, algo=MultiArmedBandit)
#guy2 = getavg(no=5, algo=MultiArmedBandit)
guy3 = getavg(no=10, algo=QLearning, epsilon=0.01)

print("group2")

#q1 = getavg(no=1, algo=QLearning)
#q5 = getavg(no=5, algo=QLearning)
q10 = getavg(no=10, algo=QLearning, epsilon=0.5)

print("group3")

#q1 = getavg(no=1, algo=QLearning)
#q5 = getavg(no=5, algo=QLearning)
q11 = getavg(no=10, algo=QLearning, epsilon=0.5, adap=True)

fig, ax = plt.subplots()

#ax.plot(lrl(guy1), guy1, c='b', label="MAB-1")
#ax.plot(lrl(guy2), guy2, c='r', label="MAB-5")
ax.plot(lrl(guy3), guy3, c='g', label="Epsilon=0.01")

#ax.plot(lrl(q1), q1, c='y', label="QL-1")
#ax.plot(lrl(q5), q5, c='k', label="QL-5")
ax.plot(lrl(q10), q10, c='m', label="Epsilon=0.5")

ax.plot(lrl(q11), q11, c='b', label="Adaptive epsilon")

plt.legend(loc="lower right")
ax.set(xlabel='Index', ylabel='Rewards',
       title='Adaptive?')
ax.grid()

fig.savefig("3B.png")
plt.show()