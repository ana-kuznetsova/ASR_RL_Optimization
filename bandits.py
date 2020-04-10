class Bandit:
    def __init__(self, tasks, batch_size):
        self.actions = np.arange(len(tasks))
        self.tasks = tasks
        self._qfunc = {}
        self.policy = {}
        self.reward_hist = []
        self.batch_size = batch_size
        
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
            l = L0-L1
            
        self.reward_hist.append(L)
        
        ##Scale reward
        q_lo = np.ceil(np.quantile(self.reward_hist, 0.2))
        q_hi = np.ceil(np.quantile(self.reward_hist, 0.8))
        
        if L < q_lo:
            return -1
        elif L > q_hi:
            return 1
        else:
            return (2*(r-q_lo)/(q_hi-q_lo))-1  
        
    def sample_task(self, prev_task_ind, same=False):
        if same:
            batch = np.random.choice(self.tasks[prev_task_ind], self.batch_size, replace=False)            
        else:
            task_ind = np.random.choice(self.num_tasks, 1)
            batch = np.random.choice(self.tasks[task_ind], self.batch_size, replace=False)     



def UCB1(data):
    pass
            