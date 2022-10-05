#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from hashlib import new
import json
import logging
import os


import numpy as np
from sklearn.utils import column_or_1d
import torch

from torch.utils.data import DataLoader
from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator
import process_new_triples as inc_p

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--use_adadelta_optim', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None,
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--adding_data', action='store_true', help='declaring that the training dataset is added and is not the first one')
    
    parser.add_argument('-data_path_train', '--data_path_train', default="data/WN18RR_inc/train1.txt", type=str)
    parser.add_argument('-data_path_old_train', '--data_path_old_train', default="", type=str)
    parser.add_argument('-data_path_entities', '--data_path_entities', default="data/WN18RR_inc/entity2id.txt", type=str)
    parser.add_argument('-data_path_rels', '--data_path_rels', default="data/WN18RR_inc/relation2id.txt", type=str)
    parser.add_argument('-train_strategy','--train_strategy', default=1, type=int,help='1: default all random')
    # Strategies to select new triple classes:
    # 1. New data and old data mixed randomly (train all data randomly and entirely without a strategy) 
    # 2. Both the head and tail new triples first. Then half new data(only head or tail) then, then old data. 
    # 3. only a head or a  tail new first, then both new data, then old data
    # 4. New and half new entity triples first(mixed randomly). Then old data. Again new data

    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('--triples_are_mapped', action='store_true')

    parser.add_argument('-node_feat_path', type=str, default=None)

    parser.add_argument('-c0', '--centrality_multiplier0', default=2.0, type=float)
    parser.add_argument('-c1', '--centrality_multiplier1', default=2.0, type=float)
    parser.add_argument('-c2', '--centrality_multiplier2', default=2.0, type=float)
    parser.add_argument('-c3', '--centrality_multiplier3', default=2.0, type=float)
    parser.add_argument('-c4', '--centrality_multiplier4', default=2.0, type=float)

    parser.add_argument('-psi', '--psi', default=14.0, type=float)

    parser.add_argument('--mde_score', action='store_true')
    parser.add_argument('-gamma_1', '--gamma_1', default=2, type=int)
    parser.add_argument('-gamma_2', '--gamma_2', default=2, type=int)
    parser.add_argument('-beta_1', '--beta_1', default=1, type=int)
    parser.add_argument('-beta_2', '--beta_2', default=1, type=int)

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--anchor_node_num', type=int, default=1, help='anchor nodes number')

    return parser.parse_args(args)


def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def read_mapped_triples(file_path, args):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id_mapped = dict()
        new_eid = 0
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id_mapped[eid] = int(new_eid)
            new_eid = new_eid + 1

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id_mapped = dict()
        new_rid = 0
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id_mapped[rid] = int(new_rid)
            new_rid = new_rid + 1

    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((int(entity2id_mapped[h]), int(relation2id_mapped[r]), int(entity2id_mapped[t])))
    return triples, entity2id_mapped, relation2id_mapped


def array_norm(array):
    max_dg = torch.max(array).item()
    min_dg = torch.min(array).item()
    if min_dg < 0:
        min_dg = 0
    out_array = (array - min_dg) / (max_dg - min_dg)
    return out_array


def get_node_features(args, entity2id):
    _degree = torch.ones(len(
        entity2id)) * 10  # this is to include nodes that are not in train. but the entity.dic must be sorted and include train nodes first.
    _degree_ = np.load(args.node_feat_path + "/degree.npy")
    max_dg = np.log((_degree_[:, 1]).astype('int').max())  # np.log(np.max(_degree_[:, 1]))
    _degree[0:_degree_.shape[0]] = torch.log(torch.tensor(_degree_[:, 1].astype('int'), dtype=float)) / max_dg

    _pagerank = torch.ones(len(entity2id)) * 10.0
    _pagerank_ = np.load(args.node_feat_path + "/pagerank.npy", allow_pickle=True, encoding='latin1')
    _pagerank_ = np.array(list(_pagerank_.item().items()))[:, 1]
    _pagerank[0:_pagerank_.shape[0]] = array_norm(torch.tensor(_pagerank_.astype('float')))

    _centrality = torch.ones(len(entity2id)) * 10.0
    _centrality_ = np.load(args.node_feat_path + "/centrality.npy", allow_pickle=True, encoding='latin1')
    _centrality_ = np.array(list(_centrality_.item().items()))[:, 1]
    _centrality[0:_centrality_.shape[0]] = array_norm(torch.tensor(_centrality_.astype('float')))

    _betweenness = torch.ones(len(entity2id)) * 10.0
    _betweenness_ = np.load(args.node_feat_path + "/betweenness.npy", allow_pickle=True, encoding='latin1')
    _betweenness_ = np.array(list(_betweenness_.item().items()))[:, 1]
    _betweenness[0:_betweenness_.shape[0]] = array_norm(torch.tensor(_betweenness_.astype('float')))

    _katz = torch.ones(len(entity2id)) * 10.0
    _katz_ = np.load(args.node_feat_path + "/katz.npy", allow_pickle=True, encoding='latin1')
    _katz_ = np.array(list(_katz_.item().items()))[:, 1]
    _katz[0:_katz_.shape[0]] = array_norm(torch.tensor(_katz_.astype('float')))

    random_paths_lenghs_ = np.load(args.node_feat_path + "/selected_lengthpaths.npy", allow_pickle=True,
                                   encoding='latin1')
    random_paths_lenghs_ = torch.tensor(random_paths_lenghs_)

    max_path_length = torch.max(random_paths_lenghs_)  # * 1.0
    random_paths_lenghs = torch.ones(len(entity2id), random_paths_lenghs_.shape[1]) * max_path_length
    random_paths_lenghs[0:random_paths_lenghs_.shape[0], :] = random_paths_lenghs_

    random_paths_lenghs[random_paths_lenghs == -1] = max_path_length + 1.0
    random_paths_lenghs = random_paths_lenghs * 1.0
    random_paths_lenghs = random_paths_lenghs / (max_path_length + 1.0)

    return [torch.stack((_degree, _pagerank, _centrality, _betweenness, _katz), dim=1).t(), random_paths_lenghs]


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

    # with open(os.path.join(args.data_path, 'entities.dict')) as fin:
    #     entity2id = dict()
    #     for line in fin:
    #         eid, entity = line.strip().split('\t')
    #         entity2id[entity] = int(eid)
    # args.entity2id = entity2id

    # with open(os.path.join(args.data_path, 'relations.dict')) as fin:
    #     relation2id = dict()
    #     for line in fin:
    #         rid, relation = line.strip().split('\t')
    #         relation2id[relation] = int(rid)

    if args.adding_data is False:
        train_triples, entity2id, relation2id = inc_p.read_new_entities_mapped(args)
        
        triple_dic_class, triple_dic, triple_dic_t, dic_e ,dic_r = inc_p.estimate_triples_class_old(train_triples)
        #np.save(os.path.join(args.data_path, 'last_train_raw'),[train_triples, entity2id, relation2id])
        #np.save(os.path.join(args.data_path, 'last_train_triple_class'),[triple_dic_class, triple_dic, triple_dic_t, dic_e ,dic_r])
    else:
        #[train_triples0, entity2id0, relation2id0] = np.load(os.path.join(args.data_path, 'last_train_raw'))
        #[triple_dic_class0, triple_dic0, triple_dic_t0, dic_e0 ,dic_r0] = np.load(os.path.join(args.data_path, 'last_train_triple_class'))
        import copy
        args2 = copy.deepcopy(args)
        args2.data_path_train = args2.data_path_old_train
        train_triples0, entity2id0, relation2id0 = inc_p.read_new_entities_mapped(args2)
        triple_dic_class0, triple_dic0, triple_dic_t0, dic_e0 ,dic_r0 = inc_p.estimate_triples_class_old(train_triples0)
        #
        train_triples, entity2id, relation2id = inc_p.read_new_entities_mapped(args, entity2id0, relation2id0,train_triples0)
        triple_dic_class, triple_dic, triple_dic_t, dic_e ,dic_r = inc_p.estimate_triples_class(train_triples,triple_dic_class0, triple_dic0, triple_dic_t0, dic_e0 ,dic_r0)
        #np.save(os.path.join(args.data_path, 'last_train_raw'),[train_triples, entity2id, relation2id])
        #np.save(os.path.join(args.data_path, 'last_train_triple_class'),[triple_dic_class, triple_dic, triple_dic_t, dic_e ,dic_r])

    inverse_dic_tirple_class = inc_p.get_inverse_class(triple_dic_class)
    print("num of class 0 triples:", len(inverse_dic_tirple_class[0])) #from old dataset
    print("num of class 1 triples:", len(inverse_dic_tirple_class.get(1,[]))) # head or tail is in the old dataset
    print("num of class 2 triples:", len(inverse_dic_tirple_class.get(2,[]))) # none of head or tails are in the old dataset
    
    if args.train_strategy == 2 or  args.train_strategy  == 3 or args.train_strategy == 4:

        old_triples = read_triple(args.data_path_old_train, entity2id, relation2id)
        strategy_stage_list = {2:[2,1,0],3:[1,2,0],4:{5,0,5}}
        # class 0 triples: from old dataset.
        # class 1 triples: head or tail is in the old dataset
        # class 2 triples: none of head or tails are in the old dataset
        # 5 for class of triples means random mix of class 1 and 2 (their sum, the random mix is done in the iterator)
        # for strategy 1 which is the random select of all, we use the default training. 
        if args.train_strategy == 4:
            sum_1_and_2 = inverse_dic_tirple_class.get(1,[]) 
            sum_1_and_2.extend(inverse_dic_tirple_class.get(2,[]) )
            train_triples_0 = sum_1_and_2
            train_triples_1 = inverse_dic_tirple_class.get(0,[])  +  old_triples 
            train_triples_2 = sum_1_and_2
        else:
            strategy_stage_set_selection = strategy_stage_list[args.train_strategy]
            train_triples_0 = inverse_dic_tirple_class.get(strategy_stage_set_selection[0],[]) +  old_triples 
            train_triples_1 = inverse_dic_tirple_class.get(strategy_stage_set_selection[1],[])
            train_triples_2 = inverse_dic_tirple_class.get(strategy_stage_set_selection[2],[])
            
        
    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    # import glob
    # trains = glob.glob("./train*.txt")
    # print("list of train files " + trains)
    # if args.triples_are_mapped:
    #     train_triples, entity2id_new, relation2id_new = read_mapped_triples(os.path.join(args.data_path, trains[0]),
    #                                                                         args)
    #     logging.info('#train: %d' % len(train_triples))

    #     valid_triples, entity2id_mapped, relation2id_mapped = read_mapped_triples(
    #         os.path.join(args.data_path, 'valid.txt'), args)
    #     logging.info('#valid: %d' % len(valid_triples))

    #     test_triples, entity2id_mapped, relation2id_mapped = read_mapped_triples(
    #         os.path.join(args.data_path, 'test.txt'), args)
    #     logging.info('#test: %d' % len(test_triples))
    # else:
    #     train_triples = read_triple(os.path.join(args.data_path, trains[0]), entity2id, relation2id)
    #     logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))

    if args.model == 'MDE':
        node_features = get_node_features(args, entity2id)

    all_true_triples = train_triples + valid_triples + test_triples

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity, #this is new size
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )

    if args.cuda:
        kge_model = kge_model.cuda()

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.model == 'MDE':
        kge_model.set_node_features(args, node_features)
    else:
        kge_model.args = args

    if args.do_train:
        # Set training dataloader iterator

        if args.train_strategy ==1:
            train_dataloader_head = DataLoader(
                TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TrainDataset.collate_fn
            )

            train_dataloader_tail = DataLoader(
                TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TrainDataset.collate_fn
            )

            train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        elif args.train_strategy ==2 or args.train_strategy ==3 or args.train_strategy ==4:
            train_iterator_set = []
            
            strategy_stage_set_selection = strategy_stage_list[args.train_strategy]
            #train_triples_0_index = strategy_stage_set_selection[0]
            #train_triples_1_index = strategy_stage_set_selection[1]
            #train_triples_2_index = strategy_stage_set_selection[2]
            
            train_dataloader_head0 = DataLoader(
                TrainDataset(train_triples_0, nentity, nrelation, args.negative_sample_size, 'head-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TrainDataset.collate_fn
            )

            train_dataloader_tail0 = DataLoader(
                TrainDataset(train_triples_0, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TrainDataset.collate_fn
            )
            train_iterator0 = BidirectionalOneShotIterator(train_dataloader_head0, train_dataloader_tail0)
            train_iterator_set.append(train_iterator0)
            if (len(train_triples_1)> 0):
                train_dataloader_head1 = DataLoader(
                    TrainDataset(train_triples_1, nentity, nrelation, args.negative_sample_size, 'head-batch'),
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=max(1, args.cpu_num // 2),
                    collate_fn=TrainDataset.collate_fn
                )
                train_dataloader_tail1 = DataLoader(
                    TrainDataset(train_triples_1, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=max(1, args.cpu_num // 2),
                    collate_fn=TrainDataset.collate_fn
                )
                train_iterator1 = BidirectionalOneShotIterator(train_dataloader_head1, train_dataloader_tail1)
                train_iterator_set.append(train_iterator1)
            if (len(train_triples_2)> 0):
                train_dataloader_head2 = DataLoader(
                    TrainDataset(train_triples_2, nentity, nrelation, args.negative_sample_size, 'head-batch'),
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=max(1, args.cpu_num // 2),
                    collate_fn=TrainDataset.collate_fn
                )
                train_dataloader_tail2 = DataLoader(
                    TrainDataset(train_triples_2, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=max(1, args.cpu_num // 2),
                    collate_fn=TrainDataset.collate_fn
                )
                train_iterator2 = BidirectionalOneShotIterator(train_dataloader_head2, train_dataloader_tail2)
                train_iterator_set.append(train_iterator2)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        if args.use_adadelta_optim:
            optimizer = torch.optim.Adadelta(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate, weight_decay=1e-6
            )
        else:

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate
            )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        if args.model == "MDE":
            old_dim = checkpoint['model_state_dict']['relation_embedding0'].size(1)# if it want to double the dim for rotatE the base dim will based on relation
            old_entity_dim = checkpoint['model_state_dict']['entity_embedding0'].size(1)
            nentity_old = checkpoint['model_state_dict']['entity_embedding0'].size(0)
            nrelation_old = checkpoint['model_state_dict']['relation_embedding0'].size(0)
        else:
            old_dim = checkpoint['model_state_dict']['relation_embedding'].size(1)# if it want to double the dim for rotatE the base dim will based on relation
            old_entity_dim = checkpoint['model_state_dict']['entity_embedding'].size(1)
            nentity_old = checkpoint['model_state_dict']['entity_embedding'].size(0)
            nrelation_old = checkpoint['model_state_dict']['relation_embedding'].size(0)
  
        #if checkpoint['model_state_dict'] != args.model:
        #    print('loaded model is different from current model')
        kge_model_old = KGEModel(
        model_name=args.model,
        nentity=nentity_old,
        nrelation=nrelation_old,
        hidden_dim=old_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
        ) 
        kge_model_old.load_state_dict(checkpoint['model_state_dict'])
        if args.model == "MDE": # this part of code reloads embedding from last training iteration, and allow increasing entity and relation number (in entity2id.txt and relation2id.txt) and their dimension in the new run
            kge_model.entity_embedding0[0:nentity_old,0:old_entity_dim].data = kge_model_old.entity_embedding0.data
            kge_model.entity_embedding1[0:nentity_old,0:old_entity_dim].data = kge_model_old.entity_embedding1.data
            kge_model.entity_embedding3[0:nentity_old,0:old_entity_dim].data = kge_model_old.entity_embedding3.data
            kge_model.entity_embedding4[0:nentity_old,0:old_entity_dim].data = kge_model_old.entity_embedding4.data
            kge_model.entity_embedding5[0:nentity_old,0:old_entity_dim].data = kge_model_old.entity_embedding5.data
            kge_model.entity_embedding7[0:nentity_old,0:old_entity_dim].data = kge_model_old.entity_embedding7.data
            kge_model.entity_embedding8[0:nentity_old,0:old_entity_dim].data = kge_model_old.entity_embedding8.data
            kge_model.entity_embedding9[0:nentity_old,0:old_entity_dim].data = kge_model_old.entity_embedding9.data
            kge_model.entity_embedding10[0:nentity_old,0:old_entity_dim].data = kge_model_old.entity_embedding10.data
            kge_model.entity_embedding11[0:nentity_old,0:old_entity_dim].data = kge_model_old.entity_embedding11.data
            kge_model.entity_embedding12[0:nentity_old,0:old_entity_dim].data = kge_model_old.entity_embedding12.data
            kge_model.relation_embedding0[0:nrelation_old,0:old_dim].data = kge_model_old.relation_embedding0.data
            kge_model.relation_embedding1[0:nrelation_old,0:old_dim].data = kge_model_old.relation_embedding1.data
            kge_model.relation_embedding2[0:nrelation_old,0:old_dim].data = kge_model_old.relation_embedding2.data
            kge_model.relation_embedding3[0:nrelation_old,0:old_dim].data = kge_model_old.relation_embedding3.data
            kge_model.relation_embedding4[0:nrelation_old,0:old_dim].data = kge_model_old.relation_embedding4.data
            kge_model.relation_embedding5[0:nrelation_old,0:old_dim].data = kge_model_old.relation_embedding5.data
            kge_model.relation_embedding6[0:nrelation_old,0:old_dim].data = kge_model_old.relation_embedding6.data
        else:
            kge_model.entity_embedding[0:nentity_old,0:old_entity_dim].data = kge_model_old.entity_embedding.data
            kge_model.relation_embedding[0:nrelation_old,0:old_dim].data = kge_model_old.relation_embedding.data
        del kge_model_old

        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_sample_size = %d' % args.negative_sample_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    # Set valid dataloader as it would be evaluated during training

    if args.do_train:
        if current_learning_rate == 0:
            current_learning_rate = args.learning_rate

        logging.info('learning_rate = %f' % current_learning_rate)

        training_logs = []

        # Training Loop
        if args.train_strategy ==1:
            if args.adding_data:
                args.max_steps = init_step + args.max_steps 
            print("training in the 1 stage strategy number:",args.train_strategy, "running step ", init_step , "to step " ,   args.max_steps)

            for step in range(init_step, args.max_steps):

                if step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))

                    if args.use_adadelta_optim:
                        optimizer = torch.optim.Adadelta(
                            filter(lambda p: p.requires_grad, kge_model.parameters()),
                            lr=current_learning_rate, weight_decay=1e-6
                        )
                    else:
                        optimizer = torch.optim.Adam(
                            filter(lambda p: p.requires_grad, kge_model.parameters()),
                            lr=current_learning_rate
                        )
                    warm_up_steps = warm_up_steps * 3
                
                log = kge_model.train_step(kge_model, optimizer, train_iterator, args)

                training_logs.append(log)

                if step % args.save_checkpoint_steps == 0:
                    save_variable_list = {
                        'step': step,
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    save_model(kge_model, optimizer, save_variable_list, args)

                if step % args.log_steps == 0:
                    metrics = {}
                    for metric in training_logs[0].keys():
                        metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                    log_metrics('Training average', step, metrics)
                    training_logs = []

                if args.do_test and step > 15000 and step % args.valid_steps == 0:
                    logging.info('Evaluating on Test Dataset...')
                    metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
                    log_metrics('Test', step, metrics)
        elif args.train_strategy ==2 or args.train_strategy ==3 or args.train_strategy ==4:
            stage_counter = 1
            one_thid = int(args.max_steps/3)
            for stage_counter in range(1,4):
                print("stage ",str(stage_counter), " of training in 3 stage strategy number:",args.train_strategy, "running step ", init_step + (stage_counter -1) * one_thid, "to step " ,  init_step + stage_counter * one_thid)
                for step in range(init_step + (stage_counter -1) * one_thid,  init_step + stage_counter * one_thid):

                    log = kge_model.train_step(kge_model, optimizer, train_iterator_set[stage_counter -1], args)

                    training_logs.append(log)

                if step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))

                    if args.use_adadelta_optim:
                        optimizer = torch.optim.Adadelta(
                            filter(lambda p: p.requires_grad, kge_model.parameters()),
                            lr=current_learning_rate, weight_decay=1e-6
                        )
                    else:
                        optimizer = torch.optim.Adam(
                            filter(lambda p: p.requires_grad, kge_model.parameters()),
                            lr=current_learning_rate
                        )
                    warm_up_steps = warm_up_steps * 3

                    if step % args.save_checkpoint_steps == 0:
                        save_variable_list = {
                            'step': step,
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps
                        }
                        save_model(kge_model, optimizer, save_variable_list, args)

                    if step % args.log_steps == 0:
                        metrics = {}
                        for metric in training_logs[0].keys():
                            metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                        log_metrics('Training average', step, metrics)
                        training_logs = []
                    if step % args.log_steps == 0:
                        metrics = {}
                        for metric in training_logs[0].keys():
                            metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                        log_metrics('Training average', step, metrics)
                        training_logs = []

                    if args.do_test and step > 15000 and step % args.valid_steps == 0:
                        logging.info('Evaluating on Test Dataset...')
                        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
                        log_metrics('Test', step, metrics)
        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)


if __name__ == '__main__':
    main(parse_args())
