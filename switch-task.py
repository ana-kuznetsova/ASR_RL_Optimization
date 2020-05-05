from data import DataSet
from utils import clear_dirs, save_batch, load_losses, train_model
import pandas as pd
import numpy as np
import math

""""
task = [1]
For each epoch:
    1. Create a new csv with task samples.
    2. Train network
    3. Store losses upon early stopping
    4. Append new task in taskqueue. 
"""

class switch-task:
    def __intit__(self, tasks):
        self.tasks = tasks
        self.loss_hist = []
        
    def create_task(self, task_q):
        curr_data = []
        for task in task_q:
            curr_data.extend(self.tasks[task])
        np.random.shuffle(curr_data)
        save_batch(curr_data,"switch-task-train")

    def train(self, num_epoches):
        num_tasks = len(self.tasks)
        for epoch in num_epoches:
            task_q = [0]
            for task in range(1, num_tasks):
                self.create_task(task_q)
                train_model()
                loss_so_far = load_losses('switch_task')
                self.loss_hist.extend(loss_so_far)
                task_q.append(task_q[-1] + 1)
        np.save(self.loss_hist, 'loss-hist-switch-task.npy')
        

