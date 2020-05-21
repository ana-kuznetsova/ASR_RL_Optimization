import pandas as pd 
import numpy as np
import os
import json
import shutil
import pickle

def clear_dirs(mode):
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
    
    if mode=='UCB1':
        delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/1/')
        delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/2/')
        delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/3/')
        delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/4/')
        delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/5/')
        delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/6/')
        delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/7/')
        delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/8/')
        delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/9/')
        delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/10/')
        delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/main_model/')
        delete_files('/N/u/anakuzne/Carbonate/curr_learning/automated_curr/')
    elif mode=='EXP3':
        delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/main_model_exp3/')
    elif mode=='LinUCB':
        delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/main_model_lin/')
    elif mode=='SWTSK':
        delete_files('/N/slate/anakuzne/tt_ckpt_automated_curr/switch_task/')

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

def train_PG(mode):
    if mode=='UCB1':
        os.system('bash scripts/tt_train_pg.sh')
    elif mode=='EXP3':
        os.system('bash scripts/tt_train_pg_exp3.sh')
    elif mode=='LinUCB':
        os.system('bash scripts/train_linUCB_main.sh')

def train_SPG(mode):
    if mode=='UCB1':
        os.system('bash scripts/tt_train_spg.sh')
    elif mode=='EXP3':
        os.system('bash scripts/tt_train_spg_exp3.sh') 

def train_SWTSK():
    os.system('bash scripts/tt_train_sw_task.sh')

def run_validation(dir):
    command = "python /N/u/anakuzne/Carbonate/curr_learning/DeepSpeech/evaluate.py -W ignore --test_files='/N/slate/anakuzne/tatar/clips/switch-task-train.csv' --test_batch_size 64 --checkpoint_dir='/N/slate/anakuzne/tt_ckpt_automated_curr/" + dir + "/ --alphabet_config_path='/N/slate/anakuzne/tatar/tt_alphabet.txt' --test_output_file='/N/u/anakuzne/Carbonate/curr_learning/automated_curr/validation_loss.json' --lm_binary_path='/N/slate/anakuzne/tatar/tt_lm.binary' --lm_trie_path='/N/slate/anakuzne/tatar/tt_trie' --report_count 4700"
    os.system(command)

def loadValLoss():
    with open('/N/u/anakuzne/Carbonate/curr_learning/automated_curr/validation_loss.json') as f:
        loss = json.load(f)
    loss = sum([l['loss'] for l in loss])/len(loss)        
    return loss
    
def load_losses(init=False, mode='UCB1'):
    if mode == 'switch-task':
        with open('/N/u/anakuzne/Carbonate/curr_learning/automated_curr/transfer_losses.pickle'):
            loss = pickle.loads(f)
        return loss
    if mode == 'UCB1':
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
    elif mode=='EXP3':
        try:#Why?
            with open('/N/u/anakuzne/Carbonate/curr_learning/automated_curr/loss_before_exp3.json') as f:
                loss_before = json.load(f)
            L1 = sum([l['loss'] for l in loss_before])/len(loss_before)
        except FileNotFoundError:
            L1 = 800
        with open('/N/u/anakuzne/Carbonate/curr_learning/automated_curr/loss_after_exp3.json') as f:
            loss_after = json.load(f)
        L2 = sum([l['loss'] for l in loss_after])/len(loss_after)

    elif mode=='LinUCB':
        with open('/N/u/anakuzne/Carbonate/curr_learning/automated_curr/loss_before_lin.json') as f:
            loss_before = json.load(f)
        L1 = sum([l['loss'] for l in loss_before])/len(loss_before)
        with open('/N/u/anakuzne/Carbonate/curr_learning/automated_curr/loss_after_lin.json') as f:
            loss_after = json.load(f)
        L2 = sum([l['loss'] for l in loss_after])/len(loss_after)
    return [L1, L2]             