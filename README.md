# GFA-NN-INC
## Incremental Knowledge Graph Embedding

### Setup

##### 1.  Make sure the Datasets are existent in "./data"

##### 2. Before executing the training, generate pre-preocessed files by
   running: ./extract_graph_features/process.sh




## Setup for incremental training:

##### assume that data cleaning and dubplicates of entities and relations are done.
##### all the new triples have at least a head or a tail in the known trained entities.


##### for the first run:
just train on the first coming set.
save the trained model

to make it you can just run for 5 epochs.

incremental training iteration:
load the saved model with its data put it in m1
 new data in new data folder arrives-> generate graph features and dictionary files entities.dic relations.dic 
the names of coming files will be like train1.txt train2.txt etc
 read the new data:
 load them seperatedyly or putting them in one bigger train.txt file?
 in this step entity matching must be done, if they are same dedblicate and label the new ones with old ones.
then make a larger dataset, still, only train on newly come entities? or their beghbours too? or all the network? 


 make a new model name it m2 with size of m1 plus new entities in m2
 copy m1 into m2
 train on new triple t2 + triples of old dataset that have a common entity or relation with t2
 save new model and aggregated triples.  


##### research question: 
to what level of neighbours entites must be train?
can these subgraphs be trained seperatedly? 


### Dataset Generation for experimetns:
for experiments: run create_inc_dataset.py that randomly select triples from train.txt and generates several incoming train files as train1.txt ,train2.txt , ...
to run: python  create_inc_dataset.py  -data_path data/WN18RR_inc -divisions 5

## Model training: (Example run)



1. step extract features:
./extract_graph_features/process.sh


2. step run the embedding:

```
python run_incremental.py  --do_train --do_test -save ./experiments/kge_baselines_wn18rr_inc --data_path ./data/WN18RR_inc --data_path_train data/WN18RR_inc/train1.txt -data_path_entities data/WN18RR_inc/entity2id.txt -data_path_rels data/WN18RR_inc/relation2id.txt --model MDE  -n 500 -b 1000 -d 200 -g 4.0 -a 2.5 -adv -lr .0005 --max_steps 10000 --test_batch_size 2 --valid_steps 10000 --log_steps 10000 --do_valid  -node_feat_path ./data/WN18RR_inc/train_node_features --cuda -psi 14.0
```


 external step 1: data integration using entity matching and deduplicates: 
 external step 2: data processing and triple classificaiton: 

there 4 types of triples must be annoated by a 4th column:
class 0. old triple, existing in the previous training
class 1 .neghbour 1 : one of the head and tails are new, this triple can be in both of the old or the new datasets
class 2. new triple: both head tail are new


3. step extract features:
./extract_graph_features/process.sh

4. then next runs in loop for the next incoming datasets:

with --init_checkpoint to load the saved model and load new train_file:

for examplle train2.txt and new parameter:  -adding_data
```
python run_incremental.py --init_checkpoint -adding_data  --do_train --do_test -save ./experiments/kge_baselines_wn18rr_inc2 --data_path ./data/WN18RR_inc --data_path_train data/WN18RR_inc/train2.txt --model MDE  -n 500 -b 1000 -d 200 -g 4.0 -a 2.5 -adv -lr .0005 --max_steps 10000 --test_batch_size 2 --valid_steps 10000 --log_steps 10000 --do_valid  -node_feat_path ./data/WN18RR_inc/train_node_features --cuda -psi 14.0
```


