# -*- coding: utf-8 -*-

"""
Author:         Shraey Bhatia
Date:           October 2016
File:           word2vectrain.py


Gives word2vec embeddings for a corpus. The input is a directory which contains tokenised documents. 
For our problem the documents already contains ngrams(phrases) of wikipedia titles in documents created by
create_ngrams.py
Output will be a word2vec trained model.
parameters for this file are taken in main_train.py

"""

import os
import gensim
import multiprocessing
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("epochs")  # Number of training epochs
parser.add_argument("input_dir")
parser.add_argument("output_dir")
args = parser.parse_args()

# Checks if the output directory specified already exists. If it does removes it.

if os.path.isdir(args.output_dir):
    del_query = "rm -r " + args.output_dir
    os.system(del_query)

# create output directory
query = "mkdir " + args.output_dir
os.system(query)

output = args.output_dir + "/word2vec"


# A word2vec class that take document files from the directory and processes words for each sentence.
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for path, subdirs, files in os.walk(self.dirname):
            for name in files:
                temp = os.path.join(path, name)
                for line in open(temp, 'r'):
                    yield line.split()


# cores = multiprocessing.cpu_count()
cores = 8
# epochs = int(args.epochs)
epochs = 5
sentences = MySentences(args.input_dir)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Model initialization
print(cores)
model = gensim.models.Word2Vec(
    size=300,
    window=5,
    min_count=20,
    sample=0.00001,
    negative=5,
    sg=1,
    workers=cores,
)
model.build_vocab(sentences)

# Model Training
model.train(
    sentences,
    total_examples=model.corpus_count,
    epochs=epochs,
    report_delay=60.0,
)

model.save(output)
