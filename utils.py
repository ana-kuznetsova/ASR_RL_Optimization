import pandas as pd 
import numpy as np
import os
import json
import shutil


def clear_dirs():
    '''
    Cleans up directories before the next training
    '''
    def delete_files(folder):
        for filename in os.listdir(folder): 
            file_path = os.path.join(folder, filename) 
            try: 
                if os.path.isfile(file_path) or os.path.islink(file_path): 
                    os.unlink(file_path) 
                elif os.path.isdir(file_path): 
                    shutil.rmtree(file_path) 
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    

    delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/1/')
    delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/2/')
    delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/3/')
    delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/main_model/')
    delete_files('/N/u/anakuzne/Carbonate/curr_learning/automated_curr/')

def save_batch(current_batch, batch_filename):
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

    df.to_csv('/N/slate/anakuzne/tatar/clips/'+batch_filename+'.csv')

def create_model(taskID):
    os.system('bash scripts/tt_init_'+str(taskID)+'.sh') 

def initialise_model(best_greedy_a):
    best_a_path = '/N/slate/anakuzne/tt_ckpt_automated_curr/' + str(best_greedy_a+1) + '/* '
    os.system('cp -r ' + best_a_path + '/N/slate/anakuzne/tt_ckpt_automated_curr/main_model')

def train_PG():
    os.system('bash scripts/tt_train_pg.sh')

def train_SPG():
    os.system('bash scripts/tt_train_spg.sh')

 


'''
def train_SPG(sample_it=0):

    os. system('cd /N/u/anakuzne/Carbonate/curr_learning/ASR_RL_Optimization/')

    if sample_it==0:
        os.system('tt_train_spg1.script')
    elif sample_it==1:
        os.system('tt_train_spg1.script')   
'''


def load_losses(init=False):
    if init:
        #Initialize losses with approximate loss values
        #L1 = int(np.random.randint(low=600, high=700, size=1))
        L1 = 1000
    else: 
        with open('/N/u/anakuzne/Carbonate/curr_learning/automated_curr/loss_before.json') as f:
            loss_before = json.load(f)
        L1 = sum([l['loss'] for l in loss_before])/len(loss_before)
    
    with open('/N/u/anakuzne/Carbonate/curr_learning/automated_curr/loss_after.json') as f:
        loss_after = json.load(f)

    L2 = sum([l['loss'] for l in loss_after])/len(loss_after)

    return [L1, L2]