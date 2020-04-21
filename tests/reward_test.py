import numpy as np
import sys
sys.path.append('..')
from bandits import *
import numpy as np
import matplotlib.pyplot as plt 

#Dummy map of Loss After and Loss Before lists.

def reward_Scaling(X):
    scaled_reward = []
    for i,x in enumerate(X[:len(X)-1]):
        reward_hist = X[:i+1]
        q_lo = np.ceil(np.quantile(reward_hist, 0.2))
        q_hi = np.ceil(np.quantile(reward_hist, 0.8))

        if x < q_lo:
            r =  -1
        elif x > q_hi:
            r = 1
        else:
            if ((q_hi-q_lo)-1) == 0:
                r = (2*(x-q_lo))/(((q_hi-q_lo)-1)+0.0000000000001)
            else:
                r = (2*(x-q_lo))/((q_hi-q_lo)-1)
        scaled_reward.append(r)
    return scaled_reward

if __name__ == "__main__":
    x = np.random.rand(1000)
    print(x)
    r = reward_Scaling(x)
    plt.plot(r)
    plt.show()
