# coding: utf-8

from time import time
import pandas as pd
import re
from os import listdir
from os.path import isfile, join
import gc
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
from constants import *

t0 = time()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

goodids = pd.read_pickle(join(ETL_PATH, 'dewiki_good_ids.pickle'))
pattern = re.compile(r'dewiki_n')
wikipath = join(SMPL_PATH, 'dewiki')
files = sorted([f for f in listdir(wikipath) if isfile(join(wikipath, f))])

def docs_to_lists(token_series):
    return tuple(token_series.tolist())

class TaggedDocuments(object):
    def __iter__(self):
        for name in files[:2]:
            gc.collect()
            corpus = name.split('.')[0]
            print('loading', corpus)
            
            f = join(wikipath, name)
            df = pd.read_pickle(f)

            # applying the same processing to each document on each iteration 
            # is quite inefficient
            df = df[df.hash.isin(goodids.index)]
            # additional fixes (should have been earlier in the pipeline)
            mask = df.token.isin(['[', ']', '<', '>'])
            df.loc[mask, POS] = PUNCT
            # remove punctuation only for doc2vec
            df = df[df.POS != PUNCT]
            df = df.groupby([HASH], sort=False)[TOKEN].agg(docs_to_lists)
            for doc_id, doc in df.iteritems():
                # the conversion of the hash_id to str is necessary since gensim trys to allocate 
                # an array for ids of size 2^64 if int values are too big.
                yield TaggedDocument(doc, [str(doc_id)])


tagged_documents = TaggedDocuments()
gc.collect()


model = Doc2Vec(
    vector_size=300, 
    window=15, min_count=20, sample=1e-5, hs=0, dm=0, negative=5, dbow_words=1, dm_concat=0, 
    workers=4, epochs=2, seed=42
)
print('build build_vocab')
model.build_vocab(tagged_documents)
print('train')
model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs, report_delay=60)
print('save')
model.save('wiki_lemma.d2v')

t1 = int(time() - t0)
print("all done in {:02d}:{:02d}:{:02d}".format(t1//3600, (t1//60) % 60, t1 % 60))
