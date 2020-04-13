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
        
    def calc_reward(self, losses, gain='PG', prev_task_ind=None):
        
        if gain=='PG':
            L = losses[0]- losses[1]
            
        elif gain=='SPG':
            same_task_batch = self.sample_task(prev_task_ind, same=True)
            #Call training function here
            L0 = losses[0]
            L1 = train()
            L = L0-L1
            
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
        
    def sample_task(self, prev_task_ind, same=False):
        if same:
            batch = np.random.choice(self.tasks[prev_task_ind], self.batch_size, replace=False)    
            return batch
        else:
            task_ind = np.random.choice(self.num_tasks, 1)[0]
            batch = np.random.choice(self.tasks[task_ind], self.batch_size, replace=False) 
            return batch



def UCB1(dataset, num_episodes, batch_size, c=0.01):
    '''
    Params:
        dataset (object): of class DataSet
        num_episodes (int): number of episodes to play
        batch_size (int): size of the training batch
    '''
    #Initialize bandit, save past actions, save past rewards
    bandit = Bandit(dataset.tasks, batch_size)
    
    #Initialization
    #Play each of the arms once, observe the reward
    start_loss = np.random.choice(500, 1)[0]
    prev_loss = 0
    
    for i in range(len(bandit.tasks)):
        #train on the task with index i
        batch = bandit.sample_task(i, same=True)
        loss = train(batch)
        prev_loss = loss
        #Calc and rescale the reward, add to rewars history
        reward = bandit.calc_reward([start_loss, loss], gain='PG', prev_task_ind=i)
        bandit.update_qfunc(reward, i)
        
    for t in range(num_episodes):
        #Take best action, observe reward, update qfunc
        action_t = bandit.take_best_action(t, c)         
        print(f"Playing action {action_t} on time step {t}...")
        print('-----------------------------------------------')
        batch = bandit.sample_task(action_t, same=True)
        loss = train(batch)
        reward = bandit.calc_reward([prev_loss, loss], gain='PG', prev_task_ind=action_t)
        bandit.update_qfunc(reward, action_t)
        prev_loss = loss
            