# coding: utf-8
import gc
from os import listdir, makedirs
from os.path import isfile, join, exists
from time import time

import pandas as pd
from gensim.models import Word2Vec, FastText

from topic_labeling.constants import SIMPLE_PATH, ETL_PATH, TOKEN, HASH, SENT_IDX, EMB_PATH
from topic_labeling.train_utils import parse_args, EpochSaver, EpochLogger
from topic_labeling.utils import init_logging, log_args


class Sentences(object):
    """this is a memory friendly approach that streams data from disk."""
    def __init__(self, input_dir, logger, use_file_cache=False, lowercase=False):
        self.input_dir = input_dir
        self.cache_dir = join(input_dir, 'cache')
        self.logger = logger
        self.use_file_cache = use_file_cache
        self.files = sorted([f for f in listdir(input_dir) if isfile(join(input_dir, f))])
        self.cached_files = None
        self.goodids = pd.read_pickle(join(ETL_PATH, 'dewiki_good_ids.pickle'))
        self.lowercase = lowercase
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
        if self.lowercase:
            df.token = df.token.str.lower()
        df = df.groupby([HASH, SENT_IDX], sort=False)[TOKEN].agg(self.docs_to_lists)
        return df

    def init_file_cache(self):
        if not exists(self.cache_dir):
            makedirs(self.cache_dir)
        self.logger.info('inititalizing file cache in ' + self.cache_dir)
        for filename in self.files:
            ser = self.load(filename)
            ser.to_pickle(join(self.cache_dir, filename.split('.')[0] + '_cache.pickle'))
        self.cached_files = sorted(
            [f for f in listdir(self.cache_dir) if isfile(join(self.cache_dir, f))]
        )

    @staticmethod
    def docs_to_lists(token_series):
        return tuple(token_series.tolist())


def main():
    # --- argument parsing ---
    (
        model_name, epochs, min_count, cores, checkpoint_every,
        cache_in_memory, lowercase, fasttext, args
    ) = parse_args(default_model_name='w2v_default', default_epochs=100)

    # --- init logging ---
    logger = init_logging(name=model_name, basic=True, to_file=True, to_stdout=False)
    log_args(logger, args)

    input_dir = join(SIMPLE_PATH, 'dewiki')
    model_dir = join(EMB_PATH, model_name)
    if not exists(model_dir):
        makedirs(model_dir)
    logger.info('model dir: ' + model_dir)

    t0 = time()
    if cache_in_memory:
        # needs approx. 25GB of RAM
        logger.info('cache data in memory')
        sentences = [s for s in Sentences(input_dir, logger, lowercase=lowercase)]
    else:
        sentences = Sentences(input_dir, logger, use_file_cache=True, lowercase=lowercase)
    gc.collect()

    # Model initialization
    logger.info('Initializing new model')
    if fasttext:
        model = FastText(
            size=300,
            window=5,
            min_count=min_count,
            sample=1e-5,
            negative=5,
            sg=1,
            seed=42,
            iter=epochs,
            workers=cores,
            min_n=3,
            max_n=6,
        )
    else:
        model = Word2Vec(
            size=300,
            window=5,
            min_count=min_count,
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
    epoch_logger = EpochLogger(logger)

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
    model.callbacks = ()
    model.save(file_path)

    t1 = int(time() - t0)
    logger.info("all done in {:02d}:{:02d}:{:02d}".format(t1//3600, (t1//60) % 60, t1 % 60))


if __name__ == '__main__':
    main()
