from data import DataSet
import numpy as np
import pickle
from utils import *

class ContextualBandit:
    def __init__(self, tasks, batch_size, dim, num_episodes, num_timesteps):
        self.tasks = tasks
        self.stored_tasks = [i for i in self.tasks]
        self.num_actions = len(tasks)
        self.batch_size = batch_size
        self.dim = dim
        self.action_gains = {str(i):[] for i in range(len(tasks))}
        #Should be A_a^-1
        self._qfunc = {'A':{str(a):np.empty((self.dim, self.dim)) for a in range(self.num_actions)}, 
                       'b':{str(b):np.empty((self.dim, 1)) for b in range(self.num_actions)}}
        self.loss_hist = []
        self.reward_hist = np.zeros((num_episodes+1, num_timesteps+1))
        self.action_hist = []
        self.sc_reward_hist = []
        self.val_loss = []
        self.empty_tasks = [False for i in range(len(self.tasks))]
        self.theta = np.zeros((self.num_actions, dim))

    def save_val_loss(self, mode, gain_type, hist_path):
        f = open(hist_path + 'val_loss_' + mode + "_" + gain_type + '.pickle', 'wb')
        pickle.dump(self.val_loss, f)
        f.close()

    def initialise_tasks(self):
        self.stored_tasks = [[i for i in row] for row in self.tasks]
        self.empty_tasks = [False for task in self.tasks]
        
    def save_hist(self, hist_path, gain_type='PG'):
        hist_path = '/N/u/anakuzne/Carbonate/curr_learning/' + hist_path.split('/')[1]+'/'
        f = open(hist_path + 'loss_lin_' + gain_type + '.pickle', 'wb')
        pickle.dump(self.loss_hist, f)
        f.close()

        f = open(hist_path + 'actions_lin_' + gain_type + '.pickle', 'wb')
        pickle.dump(self.action_hist, f)
        f.close()

        #Save cumulative reward
        f = open(hist_path + 'cumulative_r_lin_' + gain_type + '.pickle', 'wb')
        pickle.dump(self.sc_reward_hist, f)
        f.close()

        #Calculate avg reward
        r =  np.mean(self.reward_hist, axis=0)
        np.save(hist_path + 'avg_r_lin_' + gain_type + '.npy', r)

    
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
        
    
    def calc_reward(self, losses, episode, time_step):
        '''
        Rescales reward
        Stores unscaled reward in reward_hist
        Saves loss hist to loss_hist
        '''
        print('Loss array:', losses)
        #Tau 230202 - scale factor, length of the longest input
        L = (losses[0]- losses[1])
        print('L:', L)
        self.loss_hist.append(losses[1])
        
        ##Scale reward
        ## Take quantiles for N epoch starting from 0
        print('ep:', episode, 'ts:', time_step)
        rhist_sofar = np.concatenate([np.ravel(self.reward_hist[:episode]), 
                                     self.reward_hist[episode, :time_step]], axis=0)
        print('RHIST:', rhist_sofar)
        if len(rhist_sofar) == 0:
            rhist_sofar = np.append(rhist_sofar, 0)
        q_lo = np.ceil(np.quantile(rhist_sofar, 0.2))
        print('Q Low:', q_lo)
        q_hi = np.ceil(np.quantile(rhist_sofar, 0.8))
        print('Q High:', q_hi)
        if L < q_lo:
            #if mode == 'UCB1':
            r = -1
            #if mode == 'EXP3':
                #r = 0
        elif L > q_hi:
            r = 1
        else:
            if ((q_hi-q_lo)-1) == 0:
                r = 0
            else:
                r = (2*(L-q_lo))/((q_hi-q_lo)-1)
    
        #Add to reward history
        self.reward_hist[episode][time_step] = r
        self.set_cummulative_r(r)
        return r

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
        D_a = np.empty((1, self.dim))
        for a in range(self.num_actions):
            a = str(a)
            if len(self.action_gains[a])==0:
                x_t = np.random.randint(700, size=(1, self.dim))
            elif len(self.action_gains[a])==self.dim:
                x_t = np.array(self.action_gains[a]).reshape(1, -1)
            else:
                pad = np.array(self.action_gains[a])
                pad_len = self.dim - pad.shape[0]
                x_t = np.pad(pad, (pad_len, 0), mode='maximum').reshape(1, -1)                
            #x_t = self.normalize(x_t)
            D_a = np.concatenate([D_a, x_t], axis=0)
        return D_a[1: ,  :]        
    
    def update_action_gain(self, action, gain):
        if len(self.action_gains[str(action)]) < self.dim:
            self.action_gains[str(action)].append(gain)
        else:
            self.action_gains[str(action)].pop(0)
            self.action_gains[str(action)].append(gain)
        
    def upd_action_hist(self,action):
        self.action_hist.append(action)


def LinUCB(dataset, hist_path, num_episodes, num_timesteps, batch_size, gain_type='PG'):
    alpha = 0.05    
    bandit = ContextualBandit(tasks=dataset.tasks, batch_size=64, dim=5, num_episodes=num_episodes,
            num_timesteps=num_timesteps)  
    for ep in range(num_episodes):
        print('-----------------------------------')
        print(f'LinUCB: Starting episode {ep+1}...')
        print('-----------------------------------')
        bandit.initialise_tasks()
        seen = []
        for t in range(num_timesteps):
            print(f'Timestep {t}')
            #Observe context for each arm, return feature matrix
            D_a = bandit.get_features()
            probs = []
            for action in range(bandit.num_actions):
                if action in seen:
                    q_vals = bandit.get_qvalues()
                    A_a = q_vals['A'][str(action)]
                    b = q_vals['b'][str(action)]
                else:
                    A_a = np.identity(bandit.dim)
                    b = np.zeros((bandit.dim,1))
                    bandit.update_qfunc(action, A_a, b)
                    seen.append(action)
                x_t = D_a[action].reshape(-1,1)
                theta = np.dot(A_a, b)
                root = alpha*np.sqrt(np.dot(np.dot(x_t.T, A_a), x_t))
                e = np.dot(theta.T, x_t) + root
                probs.append(float(e))
            print('Probs', probs)
            #Filter empty tasks!!!!
            print('EMPTY TASKS:', bandit.empty_tasks)
            temp_probs = [(val, i) for i,val in enumerate(probs) if not bandit.empty_tasks[i]]
            if len(temp_probs) == 0:
                break
            temp_probs = sorted(temp_probs)
            a_t = temp_probs[-1][1]
            #Take argmax
            #a_t = np.argmax(np.array(probs))
            print('Argmax action:', a_t)
            bandit.upd_action_hist(a_t)
            batch = bandit.sample_task(a_t)
            save_batch(batch, 'batch_lin')
            train_PG(mode='LinUCB')
            losses = load_losses('LinUCB')
            #losses = [700, 332]
            r = bandit.calc_reward(losses, ep, t)
            print('reward:',r)
            bandit.update_action_gain(a_t, r)
            x_t = D_a[a_t]
            q_vals = bandit.get_qvalues()
            A_t = q_vals['A'][str(action)]
            b_t = q_vals['b'][str(action)]
            A_t = A_t + np.dot(x_t, x_t.T)
            x_t = x_t.reshape(-1, 1)
            b_t = b_t.reshape(-1, 1)
            b_t = b_t + r*x_t
            bandit.update_qfunc(a_t, A_t, b_t)  
            #Save histories to plot
            bandit.save_hist(hist_path, gain_type)
            print('-----------------------------------------------')
            print('Current Q-function')
            print(bandit.get_qvalues())
            print('-----------------------------------------------')
        run_validation("LinUCB", hist_path)
        dev_err = loadValLoss(hist_path)
        bandit.val_loss.append(dev_err)
        bandit.save_val_loss("LinUCB",gain_type,hist_path)
