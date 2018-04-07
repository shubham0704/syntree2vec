"""
The purpose of this file is to train 2 models on the same data

1. Our meta-path algorithm
2. Word2Vec algorithm

"""
from __future__ import print_function
import sys, os

import multiprocessing
from time import time

from graph_builder import GraphBuilder, Args
from sentence_loader import lazy_load
import gensim.models.word2vec as w2v
from node2vec.src.model_maker import model_maker



# hyperparameters for Word2Vec
num_features = 150
min_word_count = 1
num_workers = multiprocessing.cpu_count()
context_size = 6
down_sampling = 1e-3
seed = 1

G = GraphBuilder()
sizes = [1, 2, 4, 8]
tokenized_sents = []
sents = []
for size in sizes:

    ts, s = lazy_load(chunk_size=(10240)*size) # 1048576 is 1MB
    tokenized_sents.extend(ts)
    sents.extend(s)

    print ("total number of sentences in input is: ", len(sents))
    w2v_model = w2v.Word2Vec(
                            sg=1,\
                            seed=seed,\
                            workers=num_workers,\
                            size=num_features,\
                            min_count=min_word_count,\
                            window=context_size,\
                            sample=down_sampling
                        )
    # setup metapath2vec algorithm
    G.gen_data(s, ts)
    del ts
    del s
    prev = time()
    print("\nAdding parse trees to graph")
    G.gen_giant_graph()
    print("\nDone. time taken {}".format(time()-prev))

    args = Args([None])
    args.graph = G.giant_graph
    args.tag_transition = G.tag_transition
    # check training time
    prev = time()
    model = model_maker(args, G.unique_words)
    now = time() - prev
    print("\nIt took {} time for training in prior augmented-n2v for size {} MB".format(now, size))
    
    # setup word2vec algorithm
    w2v_model.build_vocab(tokenized_sents)
    prev = time()
    w2v_model.train(tokenized_sents,total_examples=w2v_model.corpus_count,epochs=w2v_model.iter)
    now = time() - prev
    print("\nIt took {} time for training w2v model for size {} MB".format(now, size))
    model.save('./data/syncode/syncode_model_'+ str(size)+ 'MB') # syncode is a registered trademark therefore just using it as a variable
    model.save('./data/w2v/w2v_model_'+ str(size)+ 'MB')
    del w2v_model
