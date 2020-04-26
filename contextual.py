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
        self.action_gains = {str(i):[] for i in range(len(tasks))}
        #Should be A_a^-1
        self._qfunc = {'A':{str(a):np.empty() for a in range(self.num_actions)}, 
                       'b':{str(b):np.empty() for b in range(self.num_actions)}}
        self.loss_hist = []
        self.reward_hist = []
        self.action_hist = []
        self.sc_reward_hist = []
        self.theta = np.zeros((self.num_actions, dim))
        
    def save_sc_rhist(self, rhist_path):
        '''
        Save the history of scaled cumulative rewards
        to file 
        '''
        f = open(rhist_path, 'wb')
        pickle.dump(self.sc_reward_hist, f)

    def save_lhist(self, lhist_path):
        f = open(lhist_path, 'wb')
        pickle.dump(self.loss_hist, f)
    
    def save_action_hist(self, action_hist_path):
        f = open(action_hist_path, 'wb')
        pickle.dump(self.action_hist, f)

    
    def update_qfunc(self, action, A, b):
        self._qfunc['A'][str(action)] = np.linalg.inv(A)
        self._qfunc['b'][str(action)] = b
        
    def get_qvalues(self):
        return self._qfunc
        
    def sample_task(self, task_ind):
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
            self.set_cumulative_r(L)
            return L
        else:
            return L
        
    def set_cummulative_r(self, scaled_r):
        '''
        Store cumulative scaled reward per time step
        '''
        last_r = 0
        if len(self.sc_reward_hist):
            last_r = self.sc_reward_hist[-1]        
        self.sc_reward_hist.append(scaled_r + last_r)
            
    def get_features(self):
        '''
        Gets the feature matrix D_a
        '''
        D_a = np.empty()
        for a in range(self.num_actions):
            if len(self.action_gains[a])==0:
                x_t = np.zeros(self.dim)
            elif len(self.action_gains[a])==self.dim:
                x_t = np.array(self.action_gains[a])
            else:
                #pad with max loss value
                x_t = np.array(self.action_gains[a])
                pad = self.dim - len(x_t)
                x_t = np.pad(x_t, (pad, 0), mode='mean')
            np.concatenate([D_a, x_t], axis=0)
        return D_a        
    
    def update_action_gain(self, action, gain):
        if len(self.action_gains[str(action)]) < self.dim:
            self.action_gains[str(action)].append(gain)
        else:
            self.action_gains[str(action)].pop(0)
            self.action_gains[str(action)].append(gain)
        
    def upd_action_hist(self,action):
        self.action_hist.append(action)
            
    def actionSeen(self, action):
        if action in self.action_hist:
            return True
        else:
            return False



def LinUCB(dataset, csv, num_episodes, num_timesteps, batch_size, gain_type='PG'):
    alpha = 0.05
    
    bandit = ContextualBandit(tasks=dataset.tasks, batch_size=64, dim=5)  
    
    for ep in range(num_episodes):
        print(f'LinUCB: Starting episode {ep+1}...')
        
        for t in num_timesteps:
            #Observe context for each arm, return feature matrix
            D_a = bandit.get_features()
            probs = []
            
            for action in range(bandit.num_actions):
                
                #If action not seen before, initialize A_a, b
                if not bandit.actionSeen(action):
                    A_a = np.identity(bandit.dim)
                    b = np.zeros(bandit.dim,1)
                
                
                q_vals = bandit.get_qvalues()
                A_a = q_vals['A'][str(action)]
                b = q_vals['b'][str(action)]
                
                theta = np.dot(A_a, b)
                
                root = alpha*np.sqrt(np.dot(np.dot(x_t.T, A_a), x_t))
                e = np.dot(theta.T, x_t) + root
                probs.append(e)
                
            #Take argmax
            a_t = np.argmax(np.array(probs))
            bandit.upd_action_hist(a_t)
            
            batch = bandit.sample_task(a_t)
            save_batch(batch, 'batch_lin')
            
            train_PG(mode='LinUCB')
            losses = load_losses_lin(feed=False)
            r = bandit.calc_raw_reward(losses, feed=False)
            bandit.update_action_gain(a_t, r)

            D_a = bandit.get_features()
            x_t = D_a[str(a_t)]

            q_vals = bandit.get_qvalues()
            A_t = q_vals['A'][str(action)]
            b_t = q_vals['b'][str(action)]
            
            A_t = A_t + np.dot(x_t, x_t.T)
            b_t = b_t + np.multiply(r, x_t)
            bandit.update_qfunc(a_t, A_t, b_t)  
            
            
            #Save histories to plot
            bandit.save_sc_rhist('sc_reward_hist_LinUCB.pickle')
            bandit.save_lhist('loss_hist_LinUCB.pickle')
            bandit.save_action_hist('action_hist_LinUCB.pickle')
            print('-----------------------------------------------')
            print('Current Q-function')
            bandit.print_qfunc()
            print('-----------------------------------------------')