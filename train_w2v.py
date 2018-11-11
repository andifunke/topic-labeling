# coding: utf-8
import sys
from os import listdir, makedirs
from os.path import isfile, join, dirname, exists
import gc
import multiprocessing
from time import time
import pandas as pd
import gensim
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
from constants import SMPL_PATH, ETL_PATH, TOKEN, POS, PUNCT, HASH, SENT_IDX
import logging
import argparse


class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training"""
    def __init__(self):
        self.epoch = 1

    def on_epoch_begin(self, model):
        print("Epoch #{:02d} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{:02d} end".format(self.epoch))
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
            model.save(filepath)
        self.epoch += 1


class Sentences(object):
    """this is a memory friendly approach that streams data from disk."""
    def __init__(self, input_dir, logger, use_file_cache=False):
        self.input_dir = input_dir
        self.cache_dir = join(input_dir, 'cache')
        self.logger = logger
        self.use_file_cache = use_file_cache
        self.files = sorted([f for f in listdir(input_dir) if isfile(join(input_dir, f))])
        self.cached_files = None
        self.goodids = pd.read_pickle(join(ETL_PATH, 'dewiki_good_ids.pickle'))
        if use_file_cache:
            self.init_file_cache()

    def __iter__(self):
        if self.use_file_cache:
            for filename in self.cached_files:
                gc.collect()
                f = join(self.input_dir, 'cache', filename)
                self.logger.info(f)
                ser = pd.read_pickle(f)
                for sent in ser:
                    yield sent
        else:
            for filename in self.files:
                ser = self.load(filename)
                for sent in ser:
                    yield sent

    def load(self, filename):
        gc.collect()
        f = join(self.input_dir, filename)
        self.logger.info(f)
        df = pd.read_pickle(f)
        df = df[df.hash.isin(self.goodids.index)]
        # fixes wrong POS tagging for punctuation
        mask_punct = df[TOKEN].isin(list('[]<>/–%'))
        df.loc[mask_punct, POS] = PUNCT
        # remove punctuation only for doc2vec
        df = df[df.POS != PUNCT]
        df = df.groupby([HASH, SENT_IDX], sort=False)[TOKEN].agg(self.docs_to_lists)
        return df

    def init_file_cache(self):
        if not exists(self.cache_dir):
            makedirs(self.cache_dir)
        self.logger.info('inititalizing file cache in', self.cache_dir)
        for filename in self.files:
            ser = self.load(filename)
            ser.to_pickle(join(self.cache_dir, filename.split('.')[0] + '_cache.pickle'))
        self.cached_files = sorted(
            [f for f in listdir(self.cache_dir) if isfile(join(self.cache_dir, f))]
        )

    @staticmethod
    def docs_to_lists(token_series):
        return tuple(token_series.tolist())


def init_logging(name='', basic=True, to_stdout=False, to_file=False,
                 log_file='log.log', log_dir='../logs'):
    if basic:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
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
    return logger


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cacheinmem', dest='cache_in_memory', action='store_true', required=False)
    parser.add_argument('--no-cacheinmem', dest='cache_in_memory', action='store_false', required=False)
    parser.set_defaults(cache_in_memory=False)

    parser.add_argument("--cores", type=int, required=False)
    parser.add_argument("--epochs", type=int, required=False)
    parser.add_argument("--checkpoint_every", type=int, required=False)
    parser.add_argument("--model_name", type=str, required=False)

    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}
    return args


def main():
    # --- argument parsing ---
    args = parse_args()
    cache_in_memory = args.get('cache_in_memory')
    cores = args.get('cores', multiprocessing.cpu_count())
    epochs = args.get('epochs', 100)
    checkpoint_every = args.get('checkpoint_every', epochs//10)
    model_name = args.get('model_name', 'w2v')

    # --- init logging ---
    logger = init_logging(name='w2v', to_file=True, log_file='w2v.log')
    logger.info('pandas: ' + pd.__version__)
    logger.info('gensim: ' + gensim.__version__)
    logger.info('python: ' + sys.version)
    logger.info('cpu count: %d' % multiprocessing.cpu_count())
    logger.info('worker used: %d' % cores)
    logger.info('epochs: %d' % epochs)
    logger.info('save checkpoint every %d epochs' % checkpoint_every)
    logger.info('cache in memory: %r' % cache_in_memory)
    logger.info('model name: ' + model_name)

    input_dir = join(SMPL_PATH, 'dewiki')
    model_dir = join(ETL_PATH, 'embeddings', model_name)
    if not exists(model_dir):
        makedirs(model_dir)
    logger.info('model dir: ' + model_dir)

    t0 = time()
    if cache_in_memory:
        # needs approx. 25GB of RAM
        logger.info('cache data in memory')
        sentences = [s for s in Sentences(input_dir, logger)]
    else:
        sentences = Sentences(input_dir, logger, use_file_cache=True)
    gc.collect()

    # Model initialization
    logger.info('Initializing new model')
    model = Word2Vec(
        size=300,
        window=5,
        min_count=20,
        sample=1e-5,
        negative=5,
        sg=1,
        seed=42,
        iter=epochs,
        workers=cores,
    )
    logger.info('Building vocab')
    model.build_vocab(sentences, progress_per=100_000)

    # Model Training
    epoch_saver = EpochSaver(model_name, model_dir, checkpoint_every)
    epoch_logger = EpochLogger()

    logger.info('Training {:d} epochs'.format(epochs))
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs,
        report_delay=60,
        callbacks=[epoch_logger, epoch_saver],
    )

    # saving model
    file_path = join(model_dir, model_name)
    logger.info('Writing model to ' + file_path)
    model.save(file_path)

    t1 = int(time() - t0)
    logger.info("all done in {:02d}:{:02d}:{:02d}".format(t1//3600, (t1//60) % 60, t1 % 60))


if __name__ == '__main__':
    main()
