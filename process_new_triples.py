import os 
import argparse
import numpy as np


def parse_args_process(args=None):
    parser = argparse.ArgumentParser(
        description='Shuffling the training data in train.txt and splits it and save it to trainx.txt files.',
        usage='create_inc_dataset.py [<args>] [-h | --help]'
    )

    parser.add_argument('-data_path_train', '--data_path_train', default="data/WN18RR_inc/train1.txt", type=str)
    parser.add_argument('-data_path_entities', '--data_path_entities', default="data/WN18RR_inc/entity2id.txt", type=str)
    parser.add_argument('-data_path_rels', '--data_path_rels', default="data/WN18RR_inc/relation2id.txt", type=str)
    # example: python create_inc_dataset.py -data_path data/data-companies -divisions 30
    return parser.parse_args(args)


#makes dictionary of entites
def read_new_entities(args, existing_entities_dic= dict([]),existing_rel_dic= dict([]),existing_triples = []):
    '''
    Read triples and map them into ids.
    '''
    triples = existing_triples
    with open(os.path.join(args.data_path_entities)) as fin:
        entity2id_mapped = existing_entities_dic
        new_eid = len(existing_entities_dic)
        for line in fin:
            eid, entity = line.strip().split('\t')
            if entity2id_mapped.get(eid,None) is None:
                entity2id_mapped[eid] = int(new_eid)
                new_eid = new_eid + 1
    with open(os.path.join(args.data_path_rels)) as fin2:
        rel2id_mapped = existing_rel_dic
        new_rid = len(existing_rel_dic)
        for line in fin2:
            rid, rel = line.strip().split('\t')
            if rel2id_mapped.get(rid,None) is None:
                rel2id_mapped[rid] = int(new_rid)
                new_rid = new_rid + 1
    with open(args.data_path_train) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            #triples.append((int(entity2id_mapped[h]), int(rel2id_mapped[r]), int(entity2id_mapped[t])))
            triples.append((h, r, t))
    return triples, entity2id_mapped, rel2id_mapped

def read_new_entities_mapped(args, existing_entities_dic= dict([]),existing_rel_dic= dict([]),existing_triples = []):
    '''
    Read triples and map them into ids.
    '''
    triples = existing_triples
    with open(os.path.join(args.data_path_entities)) as fin:
        entity2id_mapped = existing_entities_dic
        new_eid = len(existing_entities_dic)
        for line in fin:
            eid, entity = line.strip().split('\t')
            if entity2id_mapped.get(eid,None) is None:
                entity2id_mapped[eid] = int(new_eid)
                new_eid = new_eid + 1
    with open(os.path.join(args.data_path_rels)) as fin2:
        rel2id_mapped = existing_rel_dic
        new_rid = len(existing_rel_dic)
        for line in fin2:
            rid, rel = line.strip().split('\t')
            if rel2id_mapped.get(rid,None) is None:
                rel2id_mapped[rid] = int(new_rid)
                new_rid = new_rid + 1
    with open(args.data_path_train) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((int(entity2id_mapped[h]), int(rel2id_mapped[r]), int(entity2id_mapped[t])))
            #triples.append((h, r, t))
    return triples, entity2id_mapped, rel2id_mapped



#'finds an entity is new or old'
def find_entity_class(file):
    return file 

#finds that triple includes old entites or new ones
def estimate_triples_class_old(triples_old ):
    #this getting a head entity gives the relation-tails conected to it 
    triple_dic = dict([])
    triple_dic_t = dict([])
    dic_e = dict([])
    dic_r = dict([])
    #[{triple,class}] # class means that includes entites that are: 0: old :1 one entity is atleast new 2: new and both entites are two hops away from old enities :3 three hops or more, -1 both are new an not connected to any one
    triple_dic_class = dict([])
    
    for triple in triples_old:
        h_item = triple_dic.get(triple[0],None)
        t_item = triple_dic.get(triple[2],None)
        dic_e[triple[0]] = 0
        dic_e[triple[2]] = 0
        dic_r[triple[1]] = 0
        triple_dic_class[(triple[0],triple[1],triple[2])] = 0
        if h_item is None:
            triple_dic[triple[0]] = [(triple[1],triple[2])]
        else:
            temp =  h_item
            temp.append([(triple[1],triple[2])])
            triple_dic[triple[0]] = temp
        
        if t_item is None:
            triple_dic_t[triple[2]] = [(triple[1],triple[0])]
        else:
            temp =  t_item
            temp.append([(triple[1],triple[0])])
            triple_dic_t[triple[2]] = temp
    

    return triple_dic_class, triple_dic, triple_dic_t, dic_e ,dic_r


def estimate_triples_class( triples_aggregated,triple_dic_class, triple_dic, triple_dic_t, dic_e ,dic_r): 
    dic_e_new = {}
    dic_r_new = {}
   
    triple_dic_class = {x: 0 for x in triple_dic_class}
    triple_dic_class0 =triple_dic_class.copy()
    dic_e = {x: 0 for x in dic_e}
    dic_r = {x: 0 for x in dic_r}
    for triple in triples_aggregated:
        h_item = triple_dic.get(triple[0],None)
        t_item = triple_dic.get(triple[2],None)

        h_e = dic_e.get(triple[0],None)
        t_e = dic_e.get(triple[2],None)
        r_e = dic_r.get(triple[1],None)
        triple_class_ = 0
        if r_e is None:
            dic_r_new[triple[1]] = 1
            triple_class_ = 1
        if h_e is None:
            if t_e is None:
                dic_e_new[triple[0]] = 2
                dic_e_new[triple[2]] = 2
                triple_class_ = 2
            else:
                dic_e_new[triple[0]] = 1
                triple_class_ = 1
        if t_e is None:       
            if h_e is not None:
                dic_e_new[triple[2]] = 1
                triple_class_ = 1

        triple_dic_class[(triple[0],triple[1],triple[2])] = triple_class_

        if h_item is None:                
            triple_dic[triple[0]] = [(triple[1],triple[2])]
        else:
            temp =  h_item
            temp.append([(triple[1],triple[2])])
            triple_dic[triple[0]] = temp
        
        if t_item is None:
            triple_dic_t[triple[2]] = [(triple[1],triple[0])]
        else:
            temp =  t_item
            temp.append([(triple[1],triple[0])])
            triple_dic_t[triple[2]] = temp

    dic_e_new.update(dic_e)
    dic_r_new.update(dic_r)
    triple_class_ = 1
    for x in triple_dic_class0: #now checking the old dataset, labeling the neighbor hop one of those entities common between the new and old dataset
        h_e = dic_e_new.get(triple[0],None)
        t_e = dic_e_new.get(triple[2],None)
        if h_e == 0 and t_e > 0:
            dic_e_new[triple[0]] = 1
            triple_dic_class[(triple[0],triple[1],triple[2])] = triple_class_
        if t_e == 0 and h_e > 0:
            dic_e_new[triple[2]] = 1
            triple_dic_class[(triple[0],triple[1],triple[2])] = triple_class_
    return triple_dic_class, triple_dic, triple_dic_t, dic_e_new ,dic_r_new


def get_inverse_class(triple_dic_class3):
    inverse_dic_tirple_class = {}
    for triple in triple_dic_class3:
        class_ = triple_dic_class3.get(triple)
        temp =  inverse_dic_tirple_class.get(class_)
        if temp is not None:
            temp.append(triple)
            inverse_dic_tirple_class[class_] = temp
        else:
            inverse_dic_tirple_class[class_] = [triple]
    return inverse_dic_tirple_class


# this test run reads triples from "data/WN18RR_inc/train1.txt" as first coming data, process it, then takes "data/WN18RR_inc/train2.txt" and  updates triple set and classes
def test_run_process_new_triples():

    args = parse_args_process()
    triples, entity2id_mapped, rel2id_mapped = read_new_entities(args)
    #print(triples)
    print(len(triples),len(entity2id_mapped), len(rel2id_mapped))

    triple_dic_class, triple_dic, triple_dic_t, dic_e ,dic_r = estimate_triples_class_old(triples)

    print(len(triple_dic_class),len(triple_dic), len(triple_dic_t),len(dic_e))
    
    print("printing class of a triple from the first dataset:")
    print(triple_dic_class[('12213635', '_member_meronym', '12218621')]) #class 0

    args.data_path_train = "data/WN18RR_inc/train2.txt"
    triples2, entity2id_mapped2, rel2id_mapped2 = read_new_entities(args, entity2id_mapped, rel2id_mapped,triples)
    print(len(triples2),len(entity2id_mapped2), len(rel2id_mapped2))
    #print(triples2)
    triple_dic_class2, triple_dic2, triple_dic_t2, dic_e2 ,dic_r2 = estimate_triples_class(triples2,triple_dic_class, triple_dic, triple_dic_t, dic_e ,dic_r)

    print(len(triple_dic_class2),len(triple_dic2), len(triple_dic_t2),len(dic_e2))
    print("printing class of two triples:")
    print(triple_dic_class2[('09999795', '_derivationally_related_form', '07475364')])# #class 2 none of the head and tail are not in train1.txt but in train2.txt
    print(triple_dic_class2[('08146593', '_hypernym', '08189659')]) #class 1 one of the head and tail are not in train1.txt but in train2.txt
    #print(triple_dic_class2)
    args.data_path_train = "data/WN18RR_inc/train3.txt"
    triples3, entity2id_mapped3, rel2id_mapped3 = read_new_entities(args, entity2id_mapped, rel2id_mapped,triples)
    print(len(triples3),len(entity2id_mapped3), len(rel2id_mapped3))

    triple_dic_class3, triple_dic3, triple_dic_t3, dic_e3 ,dic_r3 = estimate_triples_class(triples3,triple_dic_class2, triple_dic2, triple_dic_t2, dic_e2 ,dic_r2)
    print(triple_dic_class3[('09999795', '_derivationally_related_form', '07475364')]) #class 0 the entites existed in previous rounds of trainin in either train1 or train2
    print(triple_dic_class3[('08146593', '_hypernym', '08189659')])


    inverse_dic_tirple_class = get_inverse_class(triple_dic_class3)
    print("num of class 0 triples:", len(inverse_dic_tirple_class[0])) #from old dataset
    print("num of class 1 triples:",len(inverse_dic_tirple_class[1])) # head or tail is in the old dataset
    print("num of class 2 triples:",len(inverse_dic_tirple_class[2])) # none of head or tails are in the old dataset

#test_run_process_new_triples()


# makeing test file for incremental KGE 
# this run reads triples from "data/WN18RR_inc/train2.txt" as first data, process it, then takes "data/WN18RR_inc/test.txt" and exlude triples from test that are not in train2.txt by only selecting class 0 from it and stores them in a new test2.txt file
def make_test_file_for_incremental_train():
    args = parse_args_process()
    args.data_path_train = "data/WN18RR_inc/train2.txt"
    triples, entity2id_mapped, rel2id_mapped = read_new_entities(args)
    #print(triples)
    print(len(triples),len(entity2id_mapped), len(rel2id_mapped))

    triple_dic_class, triple_dic, triple_dic_t, dic_e ,dic_r = estimate_triples_class_old(triples)

    print(len(triple_dic_class),len(triple_dic), len(triple_dic_t),len(dic_e))
    
    args.data_path_train = "data/WN18RR_inc/test_all.txt"
    triples2, entity2id_mapped2, rel2id_mapped2 = read_new_entities(args, entity2id_mapped, rel2id_mapped,triples)
    print(len(triples2),len(entity2id_mapped2), len(rel2id_mapped2))
    #print(triples2)
    triple_dic_class2, triple_dic2, triple_dic_t2, dic_e2 ,dic_r2 = estimate_triples_class(triples2,triple_dic_class, triple_dic, triple_dic_t, dic_e ,dic_r)
    inverse_dic_tirple_class = get_inverse_class(triple_dic_class2)

    #now here remove the train2-triples from the set.
    final_test_set = []
    included  = inverse_dic_tirple_class[0]
    for triple in included:
        if triple_dic_class.get(triple,None) is None:
            final_test_set.append(triple)

    print("a sample from the new test",final_test_set[0])
    np.savetxt("data/WN18RR_inc/test.txt", final_test_set,delimiter= "\t",fmt='%s' )

#make_test_file_for_incremental_train()