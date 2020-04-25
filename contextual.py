from data import DataSet
import numpy as np
import pickle
from utils import *

class ConextualBandit:
    def __init__(self, tasks, batch_size):
        self.actions = np.arange(len(tasks))
        self.tasks = tasks
        self.stored_tasks = [i for i in self.tasks]
        self.num_tasks = len(tasks)
        self._qfunc = {a:{"a":0, "r":0, "val":0} for a in range(len(tasks))}
        self.policy = {}
        self.reward_hist = [] #history of scaled rewards
        self.loss_hist = []
        self.action_hist = []
        self.sc_reward_hist = []
        self.batch_size = batch_size
        self.empty_tasks = None
        self.phi = []
    
    def calc_future_context(self):
        pass
