from data import DataSet
from bandits import UCB1



'''
Args:
    num_tasks 
    trainig csv from DeepSpeech,
    num_episodes
    batch_size 
    batch_path (str):path to save training batch for DeepSpeech 
    c=0.01, 
    gain_type='SPG'
'''

def main():

    ##Create dataset from csv
    data = DataSet(df, 3)
    data.create_tasks()

    UCB1(data, num_episodes, batch_size, c)

    pass