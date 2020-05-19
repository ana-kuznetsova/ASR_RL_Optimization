from data import DataSet
import numpy as np
import pickle
from utils import *

class Bandit:
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
        self.val_loss = []
        self.W_exp3 = np.ones(len(self.tasks))
    
    def print_weights(self):
        print(self.W_exp3)

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

    
    def update_qfunc_UCB1(self, reward, action):
        '''
        The update function to be used for UCB1
        '''
        self._qfunc[action]["a"]+=1
        self._qfunc[action]["r"]+=reward
        if self._qfunc[action]["r"] == 0:
            self._qfunc[action]["val"] = self._qfunc[action]["r"]/(self._qfunc[action]["a"]+0.000000001)
        else:
            self._qfunc[action]["val"] = self._qfunc[action]["r"]/self._qfunc[action]["a"]
   
    def print_qfunc(self):
        print(self._qfunc)

    def take_greedy_action(self):
        '''
        Exclusively used for initialization
        to move checkpoint for the main model
        '''
        action_vals = [self._qfunc[action]['val'] for action in self._qfunc]
        return int(np.argmax(action_vals))

    def take_best_action(self, mode, c=0.01, time_step=None):
        action_vals = []
        best = -1
        if mode == 'UCB1':
            for action in self._qfunc:
                q = self._qfunc[action]['val'] + c*np.sqrt((np.log(time_step)/self._qfunc[action]["a"]))
                action_vals.append((q,action))
            #Store only those actions which have non empty tasks. 
            #To preserve action index we store a tuple of (q_func value, action index) in action vals.
            action_vals = [tup for tup in action_vals if not self.empty_tasks[tup[1]]]
            if len(action_vals) == 0:
                return -1 
            action_vals = sorted(action_vals)
            best = action_vals[-1][1]
    
        if mode == 'EXP3':
            #Storing all non empty tasks and sampling from them
            tasks = [i for i in range(len(self.tasks)) if not self.empty_tasks[i]]
            if len(tasks) == 0:
                return -1
            #Probabilities of all chosen tasks
            w = [self.W_exp3[i] for i in tasks]
            sum_w = sum(w)
            num_tasks = len(tasks)
            p_t = (1 - c) * np.array([i/sum_w for i in w]) + c/num_tasks 
            best = np.random.choice(tasks, 1, p = p_t)[0]  
        return best
        
    def erase_rhist(self):
        self.reward_hist = []

    def set_cummulative_r(self, scaled_r):
        '''
        Store cumulative scaled reward per time step
        '''
        last_r = 0
        if len(self.sc_reward_hist):
            last_r = self.sc_reward_hist[-1]        
        self.sc_reward_hist.append(scaled_r + last_r)
        #max_reward = max(self.sc_reward_hist)
        #self.sc_reward_hist = [i/max_reward for i in self.sc_reward_hist]
    
    def set_avg_r(self, scaled_r):
        '''
        Store average scaled reward per time step
        for EXP3 use
        '''
        avg_r = 0
        rhist_len = len(self.sc_reward_hist)
        if rhist_len > 0:
            avg_r = self.sc_reward_hist[-1]
        reward_so_far = avg_r * rhist_len
        avg_r = (reward_so_far + scaled_r)/(1 + rhist_len)    
        self.sc_reward_hist.append(avg_r)

    def calc_raw_reward(self, losses):
        '''
        Returns raw reward without rescaling
        '''
        print('Loss array:', losses)
        L = losses[0]- losses[1]
        print('L:', L)
        self.loss_hist.append(losses[1])
        print('Loss hist:', self.loss_hist)
        self.reward_hist.append(L)
        #Save reward to the hist of cumulative scaled rewards
        self.set_cummulative_r(L)
        return L
        
    def calc_reward(self, losses, mode):
        '''
        Rescales reward
        Stores unscaled reward in reward_hist
        Saves loss hist to loss_hist
        '''
        print('Loss array:', losses)
        L = losses[0]- losses[1]
        print('L:', L)
        self.loss_hist.append(losses[1])
        print('Loss hist:', self.loss_hist)
        self.reward_hist.append(L)
        ##Scale reward
        q_lo = np.ceil(np.quantile(self.reward_hist, 0.2))
        print('Q Low:', q_lo)
        q_hi = np.ceil(np.quantile(self.reward_hist, 0.8))
        print('Q High:', q_hi)
        if L < q_lo:
            if mode == 'UCB1':
                r = -1
            if mode == 'EXP3':
                r = 0
        elif L > q_hi:
            r = 1
        else:
            if ((q_hi-q_lo)-1) == 0:
                r = (2*(L-q_lo))/(((q_hi-q_lo)-1)+0.0000000000001)
            else:
                r = (2*(L-q_lo))/((q_hi-q_lo)-1)
        #Save reward to the hist of cumulative scaled rewards
        #self.set_cummulative_r(r)
        self.set_avg_r(r)
        return r

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
    
    def resample_task(self, batch, task_ind):
        '''
        This function resamples from the passed task_iD. It also makes sure that the resampled
        batch contains rows that are not already sampled for the current batch.
        '''
        resampled_batch =  np.random.choice(self.tasks[task_ind], self.batch_size*2, replace = False)
        resampled_batch = [row for row in resampled_batch if row not in batch]
        resampled_batch = np.random.choice(resampled_batch, self.batch_size, replace = False)
        return resampled_batch

    def initialise_tasks(self):
        self.stored_tasks = [[i for i in row] for row in self.tasks]
        self.empty_tasks = [False for task in self.tasks]
    
    def update_EXP3_weights(self, reward, action, c = 0.01):
        p_t = (1 - c) * (self.W_exp3/sum(self.W_exp3)) + c/self.num_tasks 
        #Reward Mapping
        feedback = [reward/p_t[i] if i == action else 0 for i in range(self.num_tasks)]
        for i in range(self.num_tasks):
            self.W_exp3[i] = self.W_exp3[i]*np.exp(c*feedback[i]/self.num_tasks)


def EXP3(dataset, csv, num_episodes, num_timesteps, batch_size, c=0.01, gain_type='PG'):
    bandit = Bandit(tasks = dataset.tasks, batch_size = batch_size)
    for ep in range(1, num_episodes+1):
        bandit.initialise_tasks()
        print('-----------------------------------------------')
        print(f"Starting episode {ep} ...")
        print('-----------------------------------------------')
        for t in range(1, num_timesteps+1):
            choice = np.random.choice([0,1], 1, p = [1 - c, c])[0]
            num_tasks = bandit.num_tasks
            #Explore
            if choice:
                #None empty tasks
                tasks = [i for i in range(num_tasks) if not bandit.empty_tasks[i]]
                num_tasks = len(tasks)
                #Break if no tasks is non empty. 
                if num_tasks == 0:
                    break
                uni_prob = [1/num_tasks for i in range(num_tasks)]
                action_t = np.random.choice(tasks, 1, p = uni_prob)[0]
            #Exploit
            else:
                #Choose action based on probabilities.
                action_t = bandit.take_best_action(mode = 'EXP3', c=c)
                #Break if no task is non empty
                if action_t == -1:
                    break
            #Train and get the reward for the above action
            batch = bandit.sample_task(action_t)
            save_batch(current_batch = batch, batch_filename = 'batch_exp3')
            if gain_type == 'PG':
                train_PG(mode='EXP3')
            if gain_type == 'SPG':
                resampled_batch = bandit.resample_task(batch, action_t)
                save_batch(current_batch = resampled_batch, batch_filename = 'resampled_batch_exp3')
                train_SPG(mode='EXP3')
            losses = load_losses(mode="EXP3")
            reward = bandit.calc_reward(losses, mode = 'EXP3')
            bandit.update_EXP3_weights(reward = reward, action = action_t, c = c)
            print('Current reward:', reward)
            #Save histories to plot
            bandit.save_sc_rhist('sc_reward_hist_EXP3.pickle')
            bandit.save_lhist('loss_hist_EXP3.pickle')
            bandit.save_action_hist('action_hist_EXP3.pickle')
            print('-----------------------------------------------')
            print('Current Weights')
            bandit.print_weights()
            print('-----------------------------------------------')

def UCB1(dataset, csv, num_episodes, num_timesteps, batch_size, c=0.01, gain_type='PG'):
    '''
    Params:
        dataset (object): of class DataSet
        csv (df): original training csv for DeepSpeech
        num_episodes (int): number of episodes to play
        num_timesteps (int): number of time steps per episode 
        batch_size (int): size of the training batch
        batch_path (str):path to save training batch for DeepSpeech
        c (float): exploration rate
        gain_type (str): progress gain
    '''
    #Initialize bandit, save past actions, save past rewards
    bandit = Bandit(tasks = dataset.tasks, batch_size = batch_size)
    ##### Initialization ######
    #Play each of the arms once, observe the reward
    for i in range(len(bandit.tasks)):
        batch = bandit.sample_task(i)
        save_batch(current_batch = batch, batch_filename = 'batch')
        create_model(i+1)
        losses = load_losses(init=True)        
        #reward = bandit.calc_reward(losses)
        reward = bandit.calc_raw_reward(losses)
        bandit.update_qfunc_UCB1(reward, i)
    '''
    At this point we generated initial losses.
    Now pick up the best action and load the model for the best action
    '''
    init_action = bandit.take_greedy_action()
    #Move best action model to the main model ckpt dir
    initialise_model(init_action)
    #Start training from that checkpoint
    for ep in range(1, num_episodes+1):
        bandit.initialise_tasks()
        print('-----------------------------------------------')
        print(f"Starting episode {ep} ...")
        print('-----------------------------------------------')
        for t in range(1, num_timesteps+1):
            #Take best action, observe reward, update qfunc
            action_t = bandit.take_best_action(time_step = t, c = c, mode = 'UCB1')         
            print(f"Playing action {action_t} on time step {t}...")
            bandit.action_hist.append(action_t)
            if action_t==-1:
                break
            batch = bandit.sample_task(action_t)
            save_batch(current_batch = batch, batch_filename = 'batch')
            if gain_type == 'PG':
                train_PG(mode='UCB1')
            if gain_type == 'SPG':
                resampled_batch = bandit.resample_task(batch, action_t)
                save_batch(current_batch = resampled_batch, batch_filename = 'resampled_batch')
                train_SPG(mode='UCB1')
            losses = load_losses(mode='UCB1')
            reward = bandit.calc_raw_reward(losses)
            print('Current reward:', reward)
            bandit.update_qfunc_UCB1(reward, action_t)
            #Save histories to plot
            bandit.save_sc_rhist('sc_reward_hist_UCB1.pickle')
            bandit.save_lhist('loss_hist_UCB1.pickle')
            bandit.save_action_hist('action_hist_UCB1.pickle')
            print('-----------------------------------------------')
            print('Current Q-function')
            bandit.print_qfunc()
            print('-----------------------------------------------')
        #Run validation after each epoch finishes
        run_validation()