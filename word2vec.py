# coding: utf-8

import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, exists
import gc
from constants import *
import multiprocessing
from gensim.models import Word2Vec
import logging
import argparse

from gensim.models.callbacks import CallbackAny2Vec


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
    def __init__(self, path_prefix, checkpoint_every=5):
        self.path_prefix = path_prefix
        self.epoch = 1
        self.checkpoint_every = checkpoint_every

    def on_epoch_end(self, model):
        output_path = '{}_epoch{}.model'.format(self.path_prefix, self.epoch)
        if self.epoch % self.checkpoint_every == 0:
            print('save checkpoint to', output_path)
            model.save(output_path)
        self.epoch += 1


class Sentences(object):
    """this is a memory friendly approach that streams data from disk."""
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.files = sorted([f for f in listdir(input_dir) if isfile(join(input_dir, f))])

    def __iter__(self):
        for name in self.files[:]:
            gc.collect()
            f = join(self.input_dir, name)
            ser = pd.read_pickle(f)
            for sent in ser:
                yield sent


def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument("input_dir")
    # parser.add_argument("output_dir")

    parser.add_argument('--retrain', dest='retrain', action='store_true', required=False)
    parser.set_defaults(retrain=False)
    parser.add_argument('--cache_in_memory', dest='cache_in_memory', action='store_true', required=False)
    parser.set_defaults(cache_in_memory=False)

    parser.add_argument("--cores", type=int, required=False)
    parser.add_argument("--epochs", type=int, required=False)
    parser.add_argument("--model_dir", type=str, required=False)
    parser.add_argument("--checkpoint_every", type=int, required=False)
    args = vars(parser.parse_args())

    retrain = args.get('retrain', False)
    cores = args.get('cores', multiprocessing.cpu_count())
    epochs = args.get('epochs', 5)
    checkpoint_every = args.get('checkpoint_every', 1)

    print('cpu count:', multiprocessing.cpu_count())
    print('worker count:', cores)
    print('epochs:', epochs)
    print('retrain:', retrain)
    print('save checkpoint every {:d} epochs'.format(checkpoint_every))

    input_dir = join(SMPL_PATH, 'dewiki/cache')
    leaf_dir = args.get('model_dir', 'w2v_lemma')
    leaf_dir = leaf_dir if leaf_dir else 'w2v_lemma'
    # in retrain mode model_dir is the path where an existing model is loaded from
    # if retrain is False a new model will be written to model_dir
    model_dir = join(ETL_PATH, 'NETL/trained_models/' + leaf_dir)
    model_file = join(model_dir, 'w2v_lemma')

    if args.get('cache_in_memory'):
        # needs approx. 25GB of RAM
        print('cache data in memory')
        sentences = [s for s in Sentences(input_dir)]
    else:
        sentences = Sentences(input_dir)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Model initialization
    if retrain:
        print('load existing model from', model_file)
        model = Word2Vec.load(model_file)
        out_dir = join(ETL_PATH, 'NETL/trained_models/w2v_lemma' + '_retrained{:02d}epochs'.format(epochs))
        out_file = join(out_dir, 'w2v_lemma')
        if not exists(out_dir):
            makedirs(out_dir)

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
            iter=epochs,
        )
        model.build_vocab(
            sentences,
            progress_per=100_000,
        )
        out_dir = model_dir
        out_file = model_file
        if not exists(out_dir):
            makedirs(out_dir)

    checkpoint_dir = join(out_dir, 'checkpoints')
    checkpoint_file = join(checkpoint_dir, 'w2v_lemma')
    if not exists(checkpoint_dir):
        makedirs(checkpoint_dir)

    # Model Training
    epoch_saver = EpochSaver(checkpoint_file)
    epoch_logger = EpochLogger()

    print('train {:d} epochs'.format(epochs))
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs,
        report_delay=60.0,
        callbacks=[epoch_logger, epoch_saver],
    )

    print('write model to', out_file)
    model.save(out_file)


if __name__ == '__main__':
    main()
