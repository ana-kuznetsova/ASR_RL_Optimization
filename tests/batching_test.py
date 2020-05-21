import numpy as np
import sys
sys.path.append('..')
from bandits import *



def test(tasks,t):
    mode = sys.argv[1]
    bandit = Bandit(tasks, t, 64, 15)

    for ep in range(1, 10):

            bandit.initialise_tasks()
            
            print('-----------------------------------------------')
            print(f"Starting episode {ep} ...")
            print('-----------------------------------------------')
            for task in range(len(bandit.stored_tasks)):
                print("Task",task,"with",len(bandit.stored_tasks[task]),"examples")

            for t in range(1, 100):
                if mode == 'UCB1':
                    #Testing batching
                    action_t = bandit.take_best_action(mode = mode, c=0.01, time_step = t)
                    batch = bandit.sample_task(action_t)
                    resample = bandit.resample_task(batch, action_t)  
                    print("Action:", action_t, "Episode:", ep, "Timestep:",t, "batch_len:", len(batch), "Resampled batch_len:", len(resample), "Empty_task_Arr:", bandit.empty_tasks)         
                    if action_t==-1:
                        break
                    #Dummy reward
                    reward = np.random.randint(10,100)
                    bandit.update_qfunc_UCB1(reward, action_t)
                if mode == 'EXP3':
                    choice = np.random.choice([0,1], 1, p = [1 - 0.01, 0.01])[0]
                    num_tasks = bandit.num_tasks
                    #Explore
                    if choice:
                        #None empty tasks
                        tasks = [i for i in range(num_tasks) if not bandit.empty_tasks[i]]
                        num_tasks = len(tasks)
                        uni_prob = [1/num_tasks for i in range(num_tasks)]
                        action_t = np.random.choice(tasks, 1, p = uni_prob)[0]
                    #Exploit
                    else:
                        #Choose action based on probabilities.
                        action_t = bandit.take_best_action(mode = 'EXP3', c=0.01)
                    #Train and get the reward for the above action
                    batch = bandit.sample_task(action_t)
                    resample = bandit.resample_task(batch, action_t)  
                    print("Action:", action_t, "Episode:", ep, "Timestep:",t, "batch_len:", len(batch), "Resampled batch_len:", len(resample), "Empty_task_Arr:", bandit.empty_tasks, "Weights:", bandit.W_exp3)    
                    if num_tasks == 0 or action_t == -1:
                        break
                    #Dummy reward
                    l1 = np.random.randint(10,100)
                    l2 = np.random.randint(10,100)
                    reward = bandit.calc_reward([l1,l2], mode = 'EXP3')
                    bandit.update_EXP3_weights(reward = reward, action = action_t, c = 0.01)
                        

if __name__ == "__main__":
    print("Batches initialized with 200 examples with batchsize of 64")
    tstep = 200*10//15
    tasks = [[i for i in range(200)] for j in range(10)]
    test(tasks,tstep)

