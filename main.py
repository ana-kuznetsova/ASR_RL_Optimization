from data import DataSet
from bandits import UCB1, EXP3
from utils import clear_dirs

import pandas as pd
import numpy as np
import math
import os
import json
import argparse



'''
Args:
    path to the df
    num_tasks 
    trainig csv from DeepSpeech,
    num_episodes
    batch_size 
    batch_path (str):path to save training batch for DeepSpeech 
    c=0.01, 
    gain_type='SPG'
    python main.py --df_path='tt_train_with_scores.csv' --num_tasks=3 --csv='train.csv' --num_episodes=1 --batch_size=64 --gain_type='PG' --c 0.5
    python main.py --mode='EXP3' --df_path='tt_train_with_scores.csv' --num_tasks=3 --csv='train.csv' --num_episodes=1 --batch_size=64 --gain_type='PG' --c 0.1 --lr 0.01 

'''

def main(args):

    #Clean up checkpoint dirs
    clear_dirs(args.mode)

    df = pd.read_csv(args.df_path)

    ##Create dataset from csv
    data = DataSet(df, args.num_tasks)
    data.create_tasks()

    num_timesteps = int(np.ceil(len(df)/args.batch_size))

    if args.mode=='UCB1':
        print('Starting UCB1...')
        
        UCB1(data, df, args.num_episodes, num_timesteps, args.batch_size, args.c, args.gain_type)

    elif args.mode=='EXP3':


        print('Starting EXP3...')
        EXP3(data, df, args.num_episodes, num_timesteps, args.batch_size, args.c, args.gain_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--df_path', type=str, help='Path to the ranked train df', required=True)
    parser.add_argument('--num_tasks', type=int, help='Number of training buckets (lvls of difficulty)', required=True)
    parser.add_argument('--csv', type=str, help='Train csv from DeepSpeech filtered and preprocessed', required=True)
    parser.add_argument('--num_episodes', type=int, help='Num epochs', required=True)
    parser.add_argument('--batch_size', type=int, help='Batch size', required=True)
    parser.add_argument('--c', type=float, help='Exploration rate')
    parser.add_argument('--gain_type', type=str, help='Gain type (Prediction Gain, Self-Prediction Gain)', required=True)
    parser.add_argument('--mode', type=str, help='Algorithms to run', required=True)
    args = parser.parse_args()
    main(args)