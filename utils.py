import pandas as pd 
import os
import json

def read_command(path):
    with open(path, 'r') as fo:
        return fo.read()

def save_batch(current_batch):
    '''
    Saves current batch to train in DeepSpeech
    Params:
        current_batch (array): batch sampled during algo execution by bandit
        df (data set): original training csv
        path (str): path to save df
    '''
    df = pd.read_csv('train.csv')
    s = [v.replace('.mp3', '.wav') for v in current_batch]
    df = df[df['wav_filename'].isin(s)]

    df.to_csv('/N/slate/anakuzne/tatar/clips/batch.csv')


def train_PG(taskID, init = False, end = False):
    if init:
        os.system('bash tt_init_'+str(taskID)+'.sh')
    elif end:
        os.system('bash tt_end_'+str(taskID)+'.sh')
    else:
        os.system('bash tt_train_pg'+str(taskID)+'.sh')

'''
def train_SPG(sample_it=0):

    os. system('cd /N/u/anakuzne/Carbonate/curr_learning/ASR_RL_Optimization/')

    if sample_it==0:
        os.system('tt_train_spg1.script')
    elif sample_it==1:
        os.system('tt_train_spg1.script')   
'''


def load_losses():
    
    with open('/N/u/anakuzne/Carbonate/curr_learning/automated_curr/loss_before.json') as f:
        loss_before = json.load(f)
    
    with open('/N/u/anakuzne/Carbonate/curr_learning/automated_curr/loss_after.json') as f:
        loss_after = json.load(f)

    L1 = sum([l['loss'] for l in loss_before])/len(loss_before)
    L2 = sum([l['loss'] for l in loss_after])/len(loss_before)

    return [L1, L2]