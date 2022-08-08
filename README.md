# GFA-NN
## Embedding Knowledge Graphs Attentive to Positional and Centrality Qualities


### Setup

##### 1.  Make sure the Datasets are existent in "./data"

##### 2. Before executing the training, generate pre-preocessed files by
   running: ./extract_graph_features/process.sh

##### 3. With the current limitation on GPU memories, we made a multi core
   version (5 gpu cores) to allow running the model on the larger
   dataset (for example biokg). This version of the model is in the
   folder: ./5_gpu_version_of_model_for_large_datasets

##### 4. Please check the paper Appendix for the best hyper-parameters.



## Setup for incremental training:

assume that data cleaning and dubplicates of entities and relations are done.
all the new triples have at least a head or a tail in the known trained entities.


for the first run:
just train on the first coming set.
save the trained model

to make it you can just run for few epochs.

incremental training iteration:
 load the saved model with its data put it in m1
 new data in new data folder arrives-> generate graph features and dictionary files entities.dic relations.dic 
 the names of coming files will be like train1.txt train2.txt etc
 read the new data:
 load them seperatedyly or putting them in one bigger train.txt file?
 in this step entity matching must be done, if they are same dedblicate and label the new ones with old ones.
 then make a larger dataset, still, only train on newly come entities? or their beghbours too? or all the network? 
 research question: 
 to what level of neighbours entites must be train?
 can these subgraphs be trained seperatedly? 

 make a new model name it m2 with size of m1 plus new entities in m2
 copy m1 into m2
 train on new triple t2 + triples of old dataset that have a common entity or relation with t2
 save new model and aggregated triples.  

### Dataset Generation for experimetns:
for experiments: run create_inc_dataset.py that randomly select triples from train.txt and generates several incoming train files as train1.txt ,train2.txt , ...
to run: python  create_inc_dataset.py  -data_path data/WN18RR_inc -divisions 5

## GFA-NN model training example run:

on WN18RR_INC


1. extract features:
./extract_graph_features/process.sh


2. generating embedding the first dataset:

```
python run_incremental.py  --do_train --do_test -save ./experiments/kge_baselines_wn18rr_inc --data_path ./data/WN18RR_inc --train_file train1.txt  --model MDE  -n 500 -b 1000 -d 200 -g 4.0 -a 2.5 -adv -lr .0005 --max_steps 3000 --test_batch_size 2 --valid_steps 3000 --log_steps 3000 --do_valid  -node_feat_path ./data/WN18RR_inc/train_node_features --cuda -psi 14.0
```

run on the second dataset:

External step : data integration using entity matching and deduplicates: 


3. step extract features for new triples:
./extract_graph_features/process.sh

4. 

Classify new triples by their new enities and run traing for the new triples:

process_new_triples.py is imported in run_incremental.py and performs triple class labeling:

there are 3 types of triples that it annoates:

class 0 triples: from old dataset
class 1 triples: head or tail is in the old dataset
class 2 triples: none of head or tails are in the old dataset

run with --init_checkpoint and --train_strategy to load the saved model and load new train_file:
--init_checkpoint loads the saved model from last stage
--train_strategy sets the strategy to condiser different triple classes.

Strategies to select new triple classes:
1. New data and old data mixed randomly (train all data randomly and entirely without a strategy) 
2. Both the head and tail new triples first. Then half new data(only head or tail) then, then old data. 
3. Only a head or a tail new first, then both new data, then old data
4. New and half new entity triples first(mixed randomly). Then old data. Again new data

training on train2.txt :

```
python run_incremental.py  --train_strategy 2 --init_checkpoint  --do_train --do_test -save ./experiments/kge_baselines_wn18rr_inc --data_path ./data/WN18RR_inc --train_file train2.txt  --model MDE  -n 500 -b 1000 -d 200 -g 4.0 -a 2.5 -adv -lr .0005 --max_steps 3000 --test_batch_size 2 --valid_steps 3000 --log_steps 3000 --do_valid  -node_feat_path ./data/WN18RR_inc/train_node_features --cuda -psi 14.0
```
 

### FAQ 
<strong>Q</strong>: How we reproduce the results of the model for the large dataset?

<strong>A</strong>: Large datasets similar to biokg require a large number of iterations.  Since the learning rate reduces during the training we do not suggest setting max_steps to a larger number, instead, we suggest storing the trained model using -save and rerunning the training iteration several times. In our evaluation it executed the training 3 times for biokg. 


<strong>Q</strong>: Is the model open for learning furthur features? 

<strong>A</strong>: Yes, simply by adding another score and a set of embedding weights to it. Please do not forget to normalize the graph features before learning them.


