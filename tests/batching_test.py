import numpy as np
import sys
sys.path.append('..')
from bandits import *



def test(tasks):
    mode = sys.argv[1]
    bandit = Bandit(tasks, 64)

    for ep in range(1, 10):

            bandit.initialise_tasks()
            
            print('-----------------------------------------------')
            print(f"Starting episode {ep} ...")
            print('-----------------------------------------------')
            for task in range(len(bandit.stored_tasks)):
                print("Task",task,"with",len(bandit.stored_tasks[task]),"examples")

            for t in range(1, 100):
                #Testing batching
                action_t = bandit.take_best_action(mode = mode, c=0.01, time_step = t)
                batch = bandit.sample_task(action_t)
                resample = bandit.resample_task(batch, action_t)  
                print("Action:", action_t, "Episode:", ep, "Timestep:",t, "batch_len:", len(batch), "Resampled batch_len:", len(resample), "Empty_task_Arr:", bandit.empty_tasks)         
                if action_t==-1:
                    break
                #Dummy reward
                reward = np.random.randint(10,100)
                if mode == 'UCB1':
                    bandit.update_qfunc_UCB1(reward, action_t)
                if mode == 'EXP3':
                    bandit.update_qfunc_EXP3(c=0.01)
                

if __name__ == "__main__":
    print("Batches initialized with 200 examples with batchsize of 64")
    tasks = [[i for i in range(200)] for j in range(3)]
    test(tasks)

