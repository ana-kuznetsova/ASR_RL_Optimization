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
        self.contexts = {str(i):[] for i in range(len(tasks))}
        self.action_losses = {str(i):[] for i in range(len(tasks))}
        self.loss_hist = []
        self.action_hist = []
        self.theta = np.zeros((self.num_actions, dim))



def get_feedback(D_a, curr_action):
    '''
    Calculates the feedback (gain) for each context of a_t
    D_a (array): feature matrix for action a_t
    '''
    feedback = []
    prev_actions = D_a[:,0]
    for prev in prev_actions:
        ##sample batch from the previous action
        ##train on prev_batch, get L1, save ckpt
        #train on current action from the prev_batch model, Get L2
        #Clean up ckpt dir
        #Store in feedback
        pass  


def LinUCB(dataset, csv, num_episodes, num_timesteps, batch_size, gain_type='PG'):

    #Algo Params
    dim = 2
    alpha = 0.05


    context_b = ContextualBandit(tasks=[np.random.rand(1, 5) for i in range(3)], batch_size=64, dim=3)

    time_steps = 10

    probs = []

    for t in range(time_steps):
        #Observe features for each action
        conexts = context_b.contexts
        for action in range(context_b.num_actions):
            if not conexts[str(action)]:
                A_a = np.identity(dim)
                b_a = np.zeros((dim, 1))
            else:
                #b is the vec of observed feedback for each context
                # Make training function with losses for 3 contexts

                D_a = conexts[str(action)]
                theta_a = np.dot(np.linalg.inv(A_a), b_a)

                #Term under the root
                m = np.dot(np.dot(D_a.T, np.linalg.inv(A_a)), D_a)

                probs[action] = (theta_a.T, D_a) + alpha*np.sqrt(m)
            
