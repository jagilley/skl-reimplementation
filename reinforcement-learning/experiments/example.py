import gym
from code import MultiArmedBandit, QLearning
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

env = gym.make('FrozenLake-v0')

def getavg(no=1, algo=MultiArmedBandit):
    lists = []
    out = pd.DataFrame()
    for i in range((no+1)):
        agent = algo()
        action_values, rewards = agent.fit(env, steps=500)
        lists.append(rewards)
    df = pd.DataFrame(lists, columns=list(range(100)))
    return list(df.mean())

def lrl(arr):
    return list(range(len(arr)))

#guy1 = getavg(no=1, algo=MultiArmedBandit)
#guy2 = getavg(no=5, algo=MultiArmedBandit)
print("mab")
guy3 = getavg(no=10, algo=MultiArmedBandit)

#q1 = getavg(no=1, algo=QLearning)
#q5 = getavg(no=5, algo=QLearning)
print("ql")
q10 = getavg(no=10, algo=QLearning)

fig, ax = plt.subplots()

#ax.plot(lrl(guy1), guy1, c='b', label="MAB-1")
#ax.plot(lrl(guy2), guy2, c='r', label="MAB-5")
ax.plot(lrl(guy3), guy3, c='g', label="MAB-10")

#ax.plot(lrl(q1), q1, c='y', label="QL-1")
#ax.plot(lrl(q5), q5, c='k', label="QL-5")
ax.plot(lrl(q10), q10, c='m', label="QL-10")

plt.legend(loc="lower right")
ax.set(xlabel='Index', ylabel='Rewards',
       title='Problem 2B')
ax.grid()

fig.savefig("2C.png")
plt.show()