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
        
    def calc_reward(self, losses, gain='PG'):
        
        if gain=='PG':
            L = losses[0]- losses[1]
        elif gain=='SPG':
            pass
            
        
        self.reward_hist.append(L)
        
    def sample_task(self, prev_task_ind, same=False):
        if same:
            batch = np.random.choice(self.tasks[prev_task_ind], self.batch_size, replace=False)            
        else:
            task_ind = np.random.choice(self.num_tasks, 1)
            batch = np.random.choice(self.tasks[task_ind], self.batch_size, replace=False)     



def UCB1(data):
    pass
            