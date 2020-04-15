from data import DataSet
from bandits import UCB1


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
   x batch_path (str):path to save training batch for DeepSpeech 
    c=0.01, 
    gain_type='SPG'
    python main.py --df_path='tt_train_with_scores.csv' --num_tasks=3 --csv='train.csv' --num_episodes=1 --batch_size=64 --gain_type='PG'
'''

def main(args):

    df = pd.read_csv(args.df_path)

    ##Create dataset from csv
    data = DataSet(df, args.num_tasks)
    data.create_tasks()

    num_timesteps = int(np.ceil(len(df)/args.batch_size))

    print('Starting UCB1...')
    UCB1(data, args.num_episodes, num_timesteps, args.batch_size, args.c, args.gain_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--df_path', type=str, help='Path to the ranked train df')
    parser.add_argument('--num_tasks', type=int, help='Number of training buckets (lvls of difficulty)')
    parser.add_argument('--csv', type=str, help='Train csv from DeepSpeech filtered and preprocessed')
    parser.add_argument('--num_episodes', type=int, help='Num epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--c', type=float, help='Exploration rate')
    parser.add_argument('--gain_type', type=str, help='Gain type (Prediction Gain, Self-Prediction Gain)')

    args = parser.parse_args()
    print(args)
    main(args)