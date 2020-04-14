from data import DataSet
import numpy as np
from utils import read_command, save_batch, train_PG, load_losses

class Bandit:
    def __init__(self, tasks, batch_size):
        self.actions = np.arange(len(tasks))
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self._qfunc = {a:{"a":0, "r":0, "val":0} for a in range(len(tasks))}
        self.policy = {}
        self.reward_hist = [] #history of unscaled rewards
        self.batch_size = batch_size
        
    
    def update_qfunc(self, reward, action):
        self._qfunc[action]["a"]+=1
        self._qfunc[action]["r"]+=reward
        if self._qfunc[action]["r"] == 0:
            self._qfunc[action]["val"] = self._qfunc[action]["r"]/(self._qfunc[action]["a"]+0.000000001)
        else:
            self._qfunc[action]["val"] = self._qfunc[action]["r"]/self._qfunc[action]["a"]
        
    def print_qfunc(self):
        print(self._qfunc)
    
    def take_best_action(self, time_step, c=0.01):
        action_vals = []
        for action in self._qfunc:
            q = self._qfunc[action]['val'] + c*np.sqrt((np.log(time_step)/self._qfunc[action]["a"]))
            action_vals.append(q)
        print('Action vals:', action_vals)
        return np.argmax(action_vals)
        
        
    def erase_rhist(self):
        self.reward_hist = []
        
    def calc_reward(self, losses):
        L = losses[0]- losses[1]
            
        self.reward_hist.append(L)
        
        ##Scale reward
        q_lo = np.ceil(np.quantile(self.reward_hist, 0.2))
        q_hi = np.ceil(np.quantile(self.reward_hist, 0.8))
        
        if L < q_lo:
            return -1
        elif L > q_hi:
            return 1
        else:
            if ((q_hi-q_lo)-1) == 0:
                return (2*(L-q_lo))/(((q_hi-q_lo)-1)+0.0000000000001)
            else:
                return (2*(L-q_lo))/((q_hi-q_lo)-1)
        
    def sample_task(self, task_ind):
        batch = np.random.choice(self.tasks[task_ind], self.batch_size, replace=False) 
        return batch

def UCB1(dataset, csv, num_episodes, num_timesteps, batch_size, batch_path, c=0.01, gain_type='SPG'):
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
    bandit = Bandit(dataset.tasks, batch_size)
    print('Created bandit...')
    
    
    ##### Initialization ######
    #Play each of the arms once, observe the reward
         
    if gain_type=='PG':

        print('Start training...')
        for ep in range(1, num_episodes+1):
            
            print('-----------------------------------------------')
            print(f"Starting episode {ep} ...")
            print('-----------------------------------------------')

            for i in range(1, len(bandit.tasks)+1):
                batch = bandit.sample_task(i, same=True)
                #Generate two random numbers to initialize loss
                losses = np.random.randint(500, size=2)
                print('Init losses:', losses)
                reward = bandit.calc_reward(losses)
                bandit.update_qfunc(reward, i)
            
            for t in range(1, num_timesteps+1):
                #Take best action, observe reward, update qfunc
                action_t = bandit.take_best_action(t, c)         
                print(f"Playing action {action_t} on time step {t}...")
                batch = bandit.sample_task(action_t)
                save_batch(batch)
                train_PG()
                losses = load_losses()
                print('Losses:', losses)
                reward = bandit.calc_reward(losses)
                bandit.update_qfunc(reward, action_t)
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