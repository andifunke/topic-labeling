# coding: utf-8
import multiprocessing as mp
import sys
from os import makedirs
from os.path import join, exists
from pprint import pformat

import pandas as pd
import gensim
from gensim.models.callbacks import CallbackAny2Vec
import logging
import argparse


class EpochLogger(CallbackAny2Vec):
    """
    Callback to log information about training.
    Not serializable -> remove before saving the model.
    """
    def __init__(self, logger):
        self.epoch = 1
        self.logger = logger

    def on_epoch_begin(self, model):
        self.logger.info("Epoch #{:02d} start".format(self.epoch))

    def on_epoch_end(self, model):
        self.logger.info("Epoch #{:02d} end".format(self.epoch))
        self.epoch += 1


class EpochSaver(CallbackAny2Vec):
    """Callback to save model after each epoch."""
    def __init__(self, model_name, directory, checkpoint_every=5):
        self.model_name = model_name
        self.directory = join(directory, 'checkpoints')
        if not exists(self.directory):
            makedirs(self.directory)
        self.epoch = 1
        self.checkpoint_every = checkpoint_every

    def on_epoch_end(self, model):
        if self.epoch % self.checkpoint_every == 0:
            file = '{}_epoch{:02d}'.format(self.model_name, self.epoch)
            filepath = join(self.directory, file)
            print('Saving checkpoint to ' + filepath)
            callbacks = model.callbacks
            model.callbacks = ()
            model.save(filepath)
            model.callbacks = callbacks
        self.epoch += 1


def init_logging(name='', basic=True, to_stdout=False, to_file=True, log_file=None, log_dir='../logs'):

    if log_file is None:
        log_file = name+'.log' if name else 'train.log'
    if basic:
        if to_file:
            if not exists(log_dir):
                makedirs(log_dir)
            file_path = join(log_dir, log_file)
            logging.basicConfig(
                filename=file_path,
                format='%(asctime)s - %(name)s - %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                level=logging.INFO
            )
        else:
            logging.basicConfig(
                format='%(asctime)s - %(name)s - %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                level=logging.INFO
            )
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        if to_file:
            # create path if necessary
            if not exists(log_dir):
                makedirs(log_dir)
            file_path = join(log_dir, log_file)
            fh = logging.FileHandler(file_path)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        if to_stdout:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    logger.info('pandas: ' + pd.__version__)
    logger.info('gensim: ' + gensim.__version__)
    logger.info('python: ' + sys.version.replace('\n', ' '))
    return logger


def parse_args(default_model_name='x2v', default_epochs=20):
    parser = argparse.ArgumentParser()

    parser.add_argument('--cacheinmem', dest='cache_in_memory', action='store_true', required=False)
    parser.add_argument('--no-cacheinmem', dest='cache_in_memory', action='store_false', required=False)
    parser.set_defaults(cache_in_memory=False)
    parser.add_argument('--lowercase', dest='lowercase', action='store_true', required=False)
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_false', required=False)
    parser.set_defaults(lowercase=False)
    parser.add_argument('--fasttext', dest='fasttext', action='store_true', required=False)
    parser.add_argument('--no-fasttext', dest='fasttext', action='store_false', required=False)
    parser.set_defaults(lowercase=False)

    parser.add_argument("--model_name", type=str, required=False, default=default_model_name)
    parser.add_argument("--epochs", type=int, required=False, default=default_epochs)
    parser.add_argument("--min_count", type=int, required=False, default=20)
    parser.add_argument("--cores", type=int, required=False, default=mp.cpu_count())
    parser.add_argument("--checkpoint_every", type=int, required=False, default=10)

    args = parser.parse_args()
    return (
        args.model_name, args.epochs, args.min_count, args.cores, args.checkpoint_every,
        args.cache_in_memory, args.lowercase, args.fasttext, args
    )


def log_args(logger, args):
    logger.info('\n' + pformat(vars(args)))
