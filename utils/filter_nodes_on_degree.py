import pandas as pd
import os 
import argparse
import numpy as np


def parse_args_process(args=None):
    parser = argparse.ArgumentParser(
        description='reads triples in training data in train.txt and filter out nodes that have a degree smaller than a threshold value.',
        usage='filter_nodes.py [<args>] [-h | --help]'
    )

    parser.add_argument('-data_path_train', '--data_path_train', default="../data/WN18RR_inc/train.txt", type=str)
    parser.add_argument('-t', '--threshold', default=5, type=int)
    # example: python filter_nodes_on_degree.py -data_path data/data-companies -threshold 5
    return parser.parse_args(args)


def read_triples(args):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    new_eid = 0
    with open(args.data_path_train) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            #triples.append((int(entity2id_mapped[h]), int(rel2id_mapped[r]), int(entity2id_mapped[t])))
            triples.append((h, r, t))
    return triples


def filter_nodes_degree_smaller_than(triple_set, degree_threshold):
    df = pd.DataFrame(triples) 
    df2=  df[0].append(df[2]).reset_index(drop=True).astype('str') 

    #df=  pd.concat(df[0],df[2]).reset_index(drop=True)
    mask = pd.DataFrame(df2.value_counts(sort=False))
    mask =  mask[mask>=5].dropna()
    entities= mask.index
    df2 = df.loc[df[0].isin(entities) & df[0].isin(entities)]
    
    return df2



args = parse_args_process()
#args.data_path_train = "data/WN18RR_inc/train2.txt"
triples = read_triples(args)
foltered_data = filter_nodes_degree_smaller_than(triples, args.threshold)

np.savetxt(args.data_path_train+'_filtered.txt', foltered_data.to_numpy(),delimiter= "\t",fmt='%s' )

