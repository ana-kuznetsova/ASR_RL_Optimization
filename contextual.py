from data import DataSet
import numpy as np
import pickle
from utils import *


class ContextualBandit:
    def __init__(self, tasks, batch_size, dim):
        self.tasks = tasks
        self.stored_tasks = [i for i in self.tasks]
        self.num_actions = len(tasks)
        self.dim = dim
        self.contexts = {str(i):[] for i in range(len(tasks))}
        self.action_losses = {str(i):[] for i in range(len(tasks))}
        self.loss_hist = []
        self.action_hist = []
        self.theta = np.zeros((self.num_actions, dim))

def sample_task(self, task_ind, replace=False):
    if replace:
        if len(self.stored_tasks[task_ind]) =class ContextualBandit:
    def __init__(self, tasks, batch_size, dim):
        self.tasks = tasks
        self.stored_tasks = [i for i in self.tasks]
        self.num_actions = len(tasks)
        self.dim = dim
        self.action_losses = {str(i):[] for i in range(len(tasks))}
        self.loss_hist = []
        self.reward_hist = []
        self.action_hist = []
        self.avg_reward_hist = []
        self.theta = np.zeros((self.num_actions, dim))
        
    def sample_task(self, task_ind, replace=False):
        if replace:
            if len(self.stored_tasks[task_ind]) == 0:
                return self.stored_tasks[task_ind]
            if len(self.stored_tasks[task_ind]) < self.batch_size:
                batch = self.stored_tasks[task_ind]
                self.stored_tasks[task_ind] = np.array([])
                self.empty_tasks[task_ind] = True
                return batch
            if len(self.stored_tasks[task_ind]) >= self.batch_size:
                batch = np.random.choice(self.stored_tasks[task_ind], self.batch_size, replace = False)
                self.stored_tasks[task_ind] = np.array([row for row in self.stored_tasks[task_ind] if row not in batch])
                return batch
        else:
            ## Use for feedback simulation
            batch = np.random.choice(self.stored_tasks[task_ind], self.batch_size, replace = True)
            return batch
        
    def calc_raw_reward(self, losses, feed=False):
        '''
        Returns raw reward without rescaling
        '''
        print('Loss array:', losses)
        L = losses[0]- losses[1]
        if not feed:
            print('L:', L)
            self.loss_hist.append(losses[1])
            print('Loss hist:', self.loss_hist)
            self.reward_hist.append(L)
            #Save reward to the hist of cumulative scaled rewards
            self.set_avg_r(L)
            return L
        else:
            return L
        
    def set_avg_r(self, scaled_r):
        '''
        Store average scaled reward per time step
        for EXP3 use
        '''
        avg_r = 0
        rhist_len = len(self.avg_reward_hist)
        if rhist_len > 0:
            avg_r = self.avg_reward_hist[-1]
        reward_so_far = avg_r * rhist_len
        avg_r = (reward_so_far + scaled_r)/(1 + rhist_len)    
        self.avg_reward_hist.append(avg_r)
            
    def get_features(self):
        '''
        Gets the feature vector for the action
        '''
        if self.action_hist[-3:]:
            D_a = np.array(self.action_hist[-3:] + self.loss_hist[-3:])
        elif len(self.action_hist[-3:])==0:
            D_a = np.zeros(6)
        else:
            #pad actions with 0 action
            actions=np.pad(np.array(self.action_hist), (2, 0), mode='constant') 
            #pad losses with max 
            losses = np.pad(np.array(self.loss_hist), (2, 0), mode='maximum')
            D_a = np.concatenate([actions, losses], axis=0)
        return D_a
        
    def upd_action_hist(self,action):
        self.action_hist.append(action)
            
    def actionSeen(self, action):
        if action in self.action_hist:
            return True
        else:
            return False



def LinUCB(dataset, csv, num_episodes, num_timesteps, batch_size, gain_type='PG'):
    dim = 6
    alpha = 0.05

    probs = []

    time_steps = 10
    for t in range(time_steps):
        #Observe features for each action
        #D_a = bandit.get_features()
        D_a = np.random.rand(1, 6)
        for action in range(bandit.num_actions):
            if not bandit.actionSeen(action):
                A_a = np.identity(dim)
                b_a = np.zeros((dim, 1))
            else:
                A_a = np.dot(D_a.T, D_a) + np.identity(dim)
                #b_a = get_feedback(D_a, action, bandit)
                b_a = np.repeat(np.array([1]), dim).reshape(-1, 1)
                
            
            #calc mean theta
            theta_mean = np.dot(np.linalg.inv(A_a), b_a)
            m = np.dot(D_a, np.dot(np.linalg.inv(A_a), D_a.T))
            probs.append(np.dot(D_a, theta_mean) + alpha*np.sqrt(m))
        print(probs)
            
        #Choose action according to ditribution p
        actions = np.arange(3)
        #action = np.random.choice(actions, 1, p = probs)[0]
                