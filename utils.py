import pandas as pd 

def save_batch(current_batch, df, path):
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
    
