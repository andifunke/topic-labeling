# coding: utf-8
import sys
from os import listdir, makedirs
from os.path import isfile, join, dirname, exists
import gc
import multiprocessing
from time import time
import pandas as pd
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from constants import ETL_PATH, SMPL_PATH, POS, PUNCT, TOKEN, HASH, TEXT
from train_w2v import EpochSaver, EpochLogger, init_logging, parse_args


class Documents(object):

    def __init__(self, input_dir, logger, lowercase=False):
        self.input_dir = input_dir
        self.logger = logger
        self.lowercase = lowercase
        self.files = sorted([f for f in listdir(input_dir) if isfile(join(input_dir, f))])
        self.goodids = pd.read_pickle(join(ETL_PATH, 'dewiki_good_ids.pickle'))
        self.titles = pd.read_pickle(join(ETL_PATH, 'dewiki_phrases_lemmatized.pickle'))
        if lowercase:
            self.titles.token = self.titles.token.str.lower()
            self.titles.text = self.titles.text.str.lower()

    def __iter__(self):
        for name in self.files[:]:
            gc.collect()
            corpus = name.split('.')[0]
            self.logger.info('loading ' + corpus)
            
            f = join(self.input_dir, name)
            df = pd.read_pickle(f)

            # applying the same processing to each document on each iteration 
            # is quite inefficient. If applicable keep TaggedDocuments in memory
            df = df[df.hash.isin(self.goodids.index)]

            # fixes wrong POS tagging for punctuation
            mask_punct = df[TOKEN].isin(list('[]<>/–%'))
            df.loc[mask_punct, POS] = PUNCT
            # remove punctuation only for doc2vec
            df = df[df.POS != PUNCT]
            if self.lowercase:
                df.token = df.token.str.lower()
            df = df.groupby([HASH], sort=False)[TOKEN].agg(self.docs_to_lists)

            for doc_id, doc in df.iteritems():
                # The conversion of the hash_id to str is necessary since gensim trys to allocate an
                # array for ids of size 2^64 if int values are too big. 2nd tag is the lemmatized token,
                # 3rd tag is the original (underscore-concatenated) title (parenthesis removed)
                yield TaggedDocument(doc, [
                    str(doc_id),
                    self.titles.loc[doc_id, TOKEN],
                    self.titles.loc[doc_id, TEXT]
                ])

    @staticmethod
    def docs_to_lists(token_series):
        return tuple(token_series.tolist())


def main():
    # --- argument parsing ---
    args = parse_args()
    cache_in_memory = args.get('cache_in_memory')
    cores = args.get('cores', multiprocessing.cpu_count())
    epochs = args.get('epochs', 20)
    checkpoint_every = args.get('checkpoint_every', 5)
    lowercase = args.get('lowercase')
    model_name = args.get('model_name', 'd2v')

    # --- init logging ---
    logger = init_logging(name='d2v', to_file=True, log_file='d2v.log')
    logger.info('pandas: ' + pd.__version__)
    logger.info('gensim: ' + gensim.__version__)
    logger.info('python: ' + sys.version)
    logger.info('cpu count: %d' % multiprocessing.cpu_count())
    logger.info('worker used: %d' % cores)
    logger.info('epochs: %d' % epochs)
    logger.info('save checkpoint every %d epochs' % checkpoint_every)
    logger.info('cache in memory: %r' % cache_in_memory)
    logger.info('lowercase: %r' % lowercase)
    logger.info('model name: ' + model_name)

    input_dir = join(SMPL_PATH, 'dewiki')
    model_dir = join(ETL_PATH, 'embeddings', model_name)
    if not exists(model_dir):
        makedirs(model_dir)
    logger.info('model dir: ' + model_dir)

    t0 = time()
    documents = Documents(input_dir=input_dir, logger=logger, lowercase=lowercase)
    if cache_in_memory:
        documents = list(documents)
    gc.collect()

    # Model initialization
    logger.info('Initializing new model')
    model = Doc2Vec(
        vector_size=300,
        window=15,
        min_count=20,
        sample=1e-5,
        negative=5,
        hs=0,
        dm=0,
        dbow_words=1,
        dm_concat=0,
        seed=42,
        epochs=epochs,
        workers=cores,
    )
    logger.info('Building vocab')
    model.build_vocab(documents)

    # Model Training
    epoch_saver = EpochSaver(model_name, model_dir, checkpoint_every)
    epoch_logger = EpochLogger()

    logger.info('Training {:d} epochs'.format(epochs))
    model.train(
        documents,
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
