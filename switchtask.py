from data import DataSet
from utils import clear_dirs, save_batch, load_losses, train_SWTSK, run_validation, loadValLoss
import pandas as pd
import numpy as np
import math


""""
task = [1]
For each task:
    1. Create a new csv with task samples.
    2. Train network
    3. Store losses upon early stopping
    4. Append new task in taskqueue. 
"""

class SWTSK:
    def __init__(self, data):
        self.tasks = data.tasks
        self.loss_hist = []
        self.val_loss_hist = []
        
    def create_task(self, task_q):
        curr_data = []
        for task in task_q:
            curr_data.extend(self.tasks[task])
        np.random.shuffle(curr_data)
        save_batch(curr_data,"switch-task-train")

    def train(self):
        num_tasks = len(self.tasks)
        task_q = [0]
        for task in range(1, num_tasks):
            self.create_task(task_q)
            train_SWTSK()
            loss_so_far = load_losses('SWTSK')
            self.loss_hist.extend(loss_so_far)
            task_q.append(task_q[-1] + 1)
            run_validation('SWTSK', '../history_9/')
            val_loss = loadValLoss('../history_9/')
            self.val_loss_hist.append(val_loss)

        np.save('val-loss-hist-switch-task.npy', np.array(self.val_loss_hist))
        

'''
c = 5, o = 10, cr = 0.5, 1-cr = 0.5
c = 1, o = 3, cr = 0.33, 1-cr = 0.66
cr = (o - c)/o

t(x) = length of the longest sequence --> scaling factor for reward

rewards :

0: [100,50,20,10]
1: [50,30, 10,5]
2: [10, 5, 0, 1]

rewards = avg_rew = [50.33, 26, 10, 3.33]


'''