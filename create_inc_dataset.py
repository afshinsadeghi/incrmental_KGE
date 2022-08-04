
import argparse
#this script shuffles the training data in train.txt and splits it and save it to trainx.txt files.
import random,os
import numpy  as np


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Shuffling the training data in train.txt and splits it and save it to trainx.txt files.',
        usage='create_inc_dataset.py [<args>] [-h | --help]'
    )

    parser.add_argument('-data_path', '--data_path', default="data/WN18RR_inc", type=str)
    parser.add_argument('-divisions', '--divisions', default=5, type=int)
    # example: python create_inc_dataset.py -data_path data/data-companies -divisions 30
    return parser.parse_args(args)

#class Args:
#     data_path = ""
#args = Args()
#args.data_path ="data/WN18RR_inc"
#triples, entity2id_mapped, relation2id_mapped = read_mapped_triples("data/WN18RR_inc/train1.txt",args)

args = parse_args()

with open(os.path.join(args.data_path, 'entities.dict')) as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)
args.entity2id = entity2id

with open(os.path.join(args.data_path, 'relations.dict')) as fin:
    relation2id = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            #translating it:
            #triples.append((entity2id[h], relation2id[r], entity2id[t]))
            triples.append((h, r, t))
    return triples

def save_triples(file_path,triples):
    '''
    Read triples and map them into ids.
    '''
    np.savetxt(file_path+".txt", triples, delimiter="\t", fmt='%s')

triples = read_triple(args.data_path+"/train.txt",entity2id,relation2id)

#print(triples[1:3])

random.shuffle(triples)
divisions = args.divisions
div = int( len(triples) / divisions)+1
print("number of train files: ", divisions)
#print(len(triples))
#print(div * divisions)
break_number = 1
#print(triples[1:3])
new_list = []
for i in range(0,len(triples)):
    if i == div * break_number -1 :
        new_list.append(triples[i])
        if break_number != div:
            save_triples(args.data_path+"/train"+str(break_number), new_list )
            new_list = []
        break_number = break_number +1
    else:
        new_list.append(triples[i])
save_triples(args.data_path+"/train"+str(break_number), new_list )
new_list = []

