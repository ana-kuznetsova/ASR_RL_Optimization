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

    
    def update_qfunc(self, reward, action):
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

    def take_best_action(self, time_step, c=0.01):
        action_vals = []
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
    
        return best
        
    def erase_rhist(self):
        self.reward_hist = []

    def set_cummulative_r(self, scaled_r):
        '''
        Store cumulative scaled reward 
        Per time step
        '''
        last_r = 0
        if len(self.sc_reward_hist):
            last_r = self.sc_reward_hist[-1]        
        self.sc_reward_hist.append(scaled_r + last_r)
        
    def calc_reward(self, losses):
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
            r =  -1
        elif L > q_hi:
            r = 1
        else:
            if ((q_hi-q_lo)-1) == 0:
                r = (2*(L-q_lo))/(((q_hi-q_lo)-1)+0.0000000000001)
            else:
                r = (2*(L-q_lo))/((q_hi-q_lo)-1)

        #Save reward to the hist of cumulative scaled rewards
        self.set_cummulative_r(r)

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

    def initialise_tasks(self):
        self.stored_tasks = [[i for i in row] for row in self.tasks]
        self.empty_tasks = [False for task in self.tasks]

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
        save_batch(batch)
        init_PG(i+1)
        losses = load_losses(init=True)        
        reward = bandit.calc_reward(losses)
        bandit.update_qfunc(reward, i)
    


    '''
    At this point we generated initial losses.
    Now pick up the best action and load the model for the best action
    '''

    init_action = bandit.take_greedy_action()
    #Move best action model to the main model ckpt dir
    init_model_PG(init_action)

    #Start training from that checkpoint

    if gain_type=='PG':


        for ep in range(1, num_episodes+1):

            bandit.initialise_tasks()
            
            print('-----------------------------------------------')
            print(f"Starting episode {ep} ...")
            print('-----------------------------------------------')
            
            for t in range(1, num_timesteps+1):
                #Take best action, observe reward, update qfunc
                action_t = bandit.take_best_action(t, c)         
                print(f"Playing action {action_t} on time step {t}...")
                if action_t==-1:
                    break
                batch = bandit.sample_task(action_t)
                save_batch(batch)
                train_PG()
                losses = load_losses()
                reward = bandit.calc_reward(losses)
                bandit.update_qfunc(reward, action_t)
                
                #Save histories to plot
                bandit.save_sc_rhist('sc_reward_hist.pickle')
                bandit.save_lhist('loss_hist.pickle')
                bandit.save_action_hist('action_hist.pickle')
                print('-----------------------------------------------')
                print('Current Q-function')
                bandit.print_qfunc()
                print('-----------------------------------------------')

    ''' 
    elif gain_type=='SPG':
        
        for t in range(num_episodes):
            #Take best action, observe reward, update qfunc
            action_t = bandit.take_best_action(t, c)         
            print(f"Playing action {action_t} on time step {t}...")
            print('-----------------------------------------------')
            batch = bandit.sample_task(action_t)
            save_batch(batch)
            train_SPG(sample_it=0)
            batch = bandit.sample_task(action_t)
            save_batch(batch)
            train_SPG(sample_it=0)
            losses = load_losses()            
            reward = bandit.calc_reward(losses)
            bandit.update_qfunc(reward, action_t)  
    '''      