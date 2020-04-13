import pandas as pd 
import os
import json

def save_batch(current_batch, df, path='/N/slate/anakuzne/tatar/clips/batch.csv'):
    '''
    Saves current batch to train in DeepSpeech
    Params:
        current_batch (array): batch sampled during algo execution by bandit
        df (data set): original training csv
        path (str): path to save df
    '''
    s = [v.replace('.mp3', '.wav') for v in current_batch]
    df = train[train['wav_filename'].isin(s)]

    df.to_csv(path)


def train(gain_type='PG'):

    os. system('cd /N/u/anakuzne/Carbonate/curr_learning/ASR_RL_Optimization/')

    if gain_type=='PG':
        os.system('sbatch tt_train_pg.script')
    elif gain_type=='SPG':
        os.system('sbatch tt_train_spg1.script')



    with open('/N/u/anakuzne/Carbonate/curr_learning/automated_curr/loss_before.json') as f:
        loss_before = json.load(f)
    
    with open('/N/u/anakuzne/Carbonate/curr_learning/automated_curr/loss_after.json') as f:
        loss_after = json.load(f)

    L1 = sum([l['loss'] for l in loss_before])/len(loss_before)
    L2 = sum([l['loss'] for l in loss_after])/len(loss_before)

    return [L1, L2]
    