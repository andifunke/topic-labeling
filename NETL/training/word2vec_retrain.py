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
import multiprocessing
from gensim.models import Word2Vec
import logging
import argparse

from gensim.models.callbacks import CallbackAny2Vec
#from gensim.test.utils import get_tmpfile

parser = argparse.ArgumentParser()
parser.add_argument("retrain")  # Number of training epochs
parser.add_argument("cores")  # Number of training epochs
parser.add_argument("epochs")  # Number of training epochs
parser.add_argument("input_dir")
parser.add_argument("output_dir")
parser.add_argument("model_dir")
args = parser.parse_args()

# create output directory
query = "mkdir " + args.output_dir
os.system(query)
out_file = args.output_dir + "/word2vec"


class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training"""
    def __init__(self):
        self.epoch = 1

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


class EpochSaver(CallbackAny2Vec):
    """Callback to save model after each epoch."""
    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_end(self, model):
        output_path = '{}_epoch{}.model'.format(self.path_prefix, self.epoch)
        model.save(output_path)
        self.epoch += 1


# A word2vec class that take document files from the directory and processes words for each sentence.
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for path, subdirs, files in os.walk(self.dirname):
            for name in files[:1]:
                temp = os.path.join(path, name)
                for line in open(temp, 'r', encoding='utf8'):
                    yield line.split()


print('cpu count:', multiprocessing.cpu_count())
cores = int(args.cores)
print('worker count:', cores)
epochs = int(args.epochs)
sentences = MySentences(args.input_dir)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

epoch_saver = EpochSaver(out_file)
epoch_logger = EpochLogger()

# Model initialization
if args.retrain == 'True':
    model_file = args.model_dir + "/word2vec"
    print('load existing model from', model_file)
    model = Word2Vec.load(model_file)
else:
    print('construct new model')
    model = Word2Vec(
        size=300,
        window=5,
        min_count=20,
        workers=cores,
        sample=0.00001,
        negative=5,
        sg=1,
        # callbacks=[epoch_logger, epoch_saver],
        iter=epochs,
    )
    model.build_vocab(sentences)

# Model Training
print('retrain {:d} epochs'.format(epochs))
model.train(
    sentences,
    total_examples=model.corpus_count,
    epochs=epochs,
    report_delay=60.0,
    callbacks=[epoch_logger, epoch_saver],
)

print('write model to', out_file)
model.save(out_file)
