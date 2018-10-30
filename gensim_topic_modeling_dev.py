# coding: utf-8

# In[1]:


# coding: utf-8
from os import listdir, makedirs
from os.path import join, isfile, isdir, exists
from typing import Dict, Any

import pandas as pd
import gc
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import CoherenceModel, TfidfModel, LdaModel, LdaMulticore
# from gensim.models.hdpmodel import HdpModel, HdpTopicFormatter
from gensim.models.callbacks import CoherenceMetric, DiffMetric, PerplexityMetric, ConvergenceMetric
from itertools import chain, islice

from gensim.models.wrappers import LdaMallet

from constants import (
    FULL_PATH, ETL_PATH, NLP_PATH, SMPL_PATH, POS, NOUN, PROPN, TOKEN, HASH, SENT_IDX, PUNCT
)
# import logging
import json
import numpy as np


# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# pd.options.display.max_rows = 2001


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
args = vars(parser.parse_args())


def docs_to_lists(token_series):
    return tuple(token_series.tolist())


def docs2corpora(documents, tfidf=True, stopwords=None, filter_below=5, filter_above=0.5,
                 split=False, max_test_size_rel=0.1, max_test_size_abs=5000):
    dictionary = Dictionary(documents)
    dictionary.filter_extremes(no_below=filter_below, no_above=filter_above)

    # filter some noice (e.g. special characters)
    if stopwords:
        stopword_ids = [dictionary.token2id[token] for token in stopwords]
        dictionary.filter_tokens(bad_ids=stopword_ids, good_ids=None)

    length = len(documents)
    corpora = dict()
    if split:
        if length * max_test_size_rel < max_test_size_abs:
            split1 = int(length * (1 - (2 * max_test_size_rel)))
            split2 = int(length * (1 - max_test_size_rel))
        else:
            split1 = length - (2 * max_test_size_abs)
            split2 = length - max_test_size_abs
        training_texts = documents[:split1]
        holdout_texts = documents[split1:split2]
        test_texts = documents[split2:]
        print(
            f'split dataset. size of:',
            f'train_set={len(training_texts)},',
            f'val_set={len(holdout_texts)},',
            f'test_set={len(test_texts)},'
        )
        corpora['training_corpus'] = [dictionary.doc2bow(text) for text in training_texts]
        corpora['holdout_corpus'] = [dictionary.doc2bow(text) for text in holdout_texts]
        corpora['test_corpus'] = [dictionary.doc2bow(text) for text in test_texts]
    else:
        training_texts = documents
        corpora['training_corpus'] = [dictionary.doc2bow(text) for text in training_texts]
        corpora['holdout_corpus'], corpora['test_corpus'] = None, None

    if tfidf:
        for key, bow_corpus in corpora.items():
            tfidf_model = TfidfModel(bow_corpus)
            corpora[key] = tfidf_model[bow_corpus]
    return corpora, dictionary


# In[8]:


datasets = {
    'E': 'Europarl',
    'FA': 'FAZ_combined',
    'FO': 'FOCUS_cleansed',
    'O': 'OnlineParticipation',
    'P': 'PoliticalSpeeches',
    'dewi': 'dewiki',
    'dewa': 'dewac',
}
goodids = {
    # filetered via some fixed rules and similarity measure to character distribution
    'dewac': join(ETL_PATH, 'dewac_good_ids.pickle'),
    'dewiki': join(ETL_PATH, 'dewiki_good_ids.pickle'),
    # the samples contain only a small subset of all articles
    # the reason for this is that the samples are roughly equal in size per category
    # 'FAZ_combined': join(ETL_PATH, 'FAZ_document_sample3.pickle'),
    # 'FOCUS_cleansed': join(ETL_PATH, 'FOCUS_document_sample3.pickle'),
}
bad_tokens = {
    'Europarl': [
        'E.', 'Kerr', 'The', 'la', 'ia', 'For', 'Ieke', 'the',
    ],
    'FAZ_combined': [
        'S.', 'j.reinecke@faz.de', 'B.',
    ],
    'FOCUS_cleansed': [],
    'OnlineParticipation': [
        'Re', '@#1', '@#2', '@#3', '@#4', '@#5', '@#6', '@#7', '@#8', '@#9', '@#1.1', 'Für', 'Muss',
        'etc', 'sorry', 'Ggf', 'u.a.', 'z.B.'
                                       'B.', 'stimmt', ';-)', 'lieber', 'o.', 'Ja', 'Desweiteren',
    ],
    'PoliticalSpeeches': [],
    'dewiki': [],
    'dewac': [],
}
all_bad_tokens = set(chain(*bad_tokens.values()))

# In[24]:


# for dataset in datasets[1:2]:
dataset = datasets[args['dataset']]
print('dataset:', dataset)
print('-' * 5)
dir_path = join(SMPL_PATH, 'wiki_phrases')
files = sorted([f for f in listdir(dir_path) if f.startswith(dataset)])
for name in files:
    full_path = join(dir_path, name)
    if isdir(full_path):
        subdir = sorted([join(name, f) for f in listdir(full_path) if f.startswith(dataset)])
        files += subdir

keepids = None
if dataset in goodids:
    keepids = pd.read_pickle(goodids[dataset])

nb_words = 0
reduction_pos_set = {NOUN, PROPN, 'NER', 'NPHRASE'}
documents = []
for name in files:
    gc.collect()
    full_path = join(dir_path, name)
    if not isfile(full_path):
        continue

    print('reading', name)
    df = pd.read_pickle(join(dir_path, name))
    print('    initial number of words:', len(df))
    if keepids is not None:
        # some datasets have already been filtered so you may not see a difference in any case
        df = df[df.hash.isin(keepids.index)]

    # fixing bad POS tagging
    mask = df.token.isin(list('[]<>/–%'))
    df.loc[mask, POS] = PUNCT

    # using only certain POS tags
    df = df[df.POS.isin(reduction_pos_set)]
    df[TOKEN] = df[TOKEN].map(lambda x: x.strip('-/'))
    df = df[df.token.str.len() > 1]
    df = df[~df.token.isin(all_bad_tokens)]
    nb_words += len(df)
    print('    remaining number of words:', len(df))

    # groupby sorts the documents by hash-id
    # which is equal to shuffeling the dataset before building the model
    df = df.groupby([HASH])[TOKEN].agg(docs_to_lists)
    print('    number of documents:', len(df))
    documents += df.values.tolist()

nb_docs = len(documents)
print('-' * 5)
print('total number of documents:', nb_docs)
print('total number of words:', nb_words)
stats = dict(dataset=dataset, pos_set=sorted(reduction_pos_set), nb_docs=nb_docs, nb_words=nb_words)
del keepids, files
gc.collect()


# In[20]:


corpora, dictionary = docs2corpora(
    documents, tfidf=False,
    # stopwords=bad_tokens[dataset],
    # stopword removale has been moved to the pandas preprocessing pipeline
    filter_below=5, filter_above=0.5,
    split=True,
)
training_corpus = corpora['training_corpus']
holdout_corpus = corpora['holdout_corpus']
test_corpus = corpora['test_corpus']


def init_callbacks(viz_env=None, title_suffix=''):
    # define perplexity callback for hold_out and test corpus
    pl_holdout = PerplexityMetric(
        corpus=holdout_corpus, logger="visdom", viz_env=viz_env,
        title="Perplexity (hold_out)" + title_suffix
    )
    pl_test = PerplexityMetric(
        corpus=test_corpus, logger="visdom", viz_env=viz_env,
        title="Perplexity (test)" + title_suffix
    )
    # define other remaining metrics available
    ch_umass = CoherenceMetric(
        corpus=training_corpus, coherence="u_mass", topn=10, logger="visdom", viz_env=viz_env,
        title="Coherence (u_mass)" + title_suffix
    )
    ch_cv = CoherenceMetric(
        corpus=training_corpus, texts=documents, coherence="c_v", topn=10, logger="visdom",
        viz_env=viz_env, title="Coherence (c_v)" + title_suffix
    )
    diff_kl = DiffMetric(
        distance="kullback_leibler", logger="visdom", viz_env=viz_env,
        title="Diff (kullback_leibler)" + title_suffix
    )
    convergence_kl = ConvergenceMetric(
        distance="jaccard", logger="visdom", viz_env=viz_env,
        title="Convergence (jaccard)" + title_suffix
    )
    return [pl_holdout, pl_test, ch_umass, ch_cv, diff_kl, convergence_kl]


# In[27]:


def get_parameterset(corpus, dictionary, callbacks=None, nbtopics=100, parametrization='a42',
                     eval_every=None):
    print(f'building LDA model "{parametrization}" with {nbtopics} number of topics')
    default = dict(random_state=42, corpus=corpus, id2word=dictionary, num_topics=nbtopics,
                   eval_every=eval_every, callbacks=callbacks, chunksize=20_000)
    ldamodels = {
        'a42': dict(passes=10),
        'b42': dict(passes=10, iterations=200),
        'c42': dict(passes=10, iterations=1_000),
        'd42': dict(passes=10, iterations=200, alpha=0.1, eta=0.01),
        'e42': dict(passes=20, iterations=200, alpha='auto', eta='auto'),
    }
    for key, dic in ldamodels.items():
        dic.update(default)
    return ldamodels[parametrization]


params_list = ['a42', 'b42', 'c42', 'd42', 'e42']
implementations = [
    ('LDAmodel', LdaModel),
    ('LDAmulticore', LdaMulticore),
    ('LDAmallet', LdaMallet),
]
choice = 0
model_name = implementations[choice][0]
UsedModel = implementations[choice][1]
save = True
metrics = []
# params = params_list[3]
for params in params_list:
    env_id = f"{dataset}-{model_name}"
    for nbtopics in [10, 25, 50, 100]:  # range(10, 101, 10):
        # Choose α from [0.05, 0.1, 0.5, 1, 5, 10]
        # Choose β from [0.05, 0.1, 0.5, 1, 5, 10]
        callbacks = init_callbacks(viz_env=env_id, title_suffix=f", {params}, {nbtopics}tpx")
        kwargs = get_parameterset(training_corpus, dictionary, callbacks=callbacks,
                                  nbtopics=nbtopics, parametrization=params)
        if 'multicore' in model_name:
            kwargs['workers'] = 3
            kwargs.pop('callbacks', None)
        ldamodel = UsedModel(**kwargs)

        topics = [[dataset] + [dictionary[term[0]] for term in ldamodel.get_topic_terms(i)] for i in
                  range(nbtopics)]
        df_lda = pd.DataFrame(topics, columns=['dataset'] + ['term' + str(i) for i in range(10)])

        current_metrics = ldamodel.metrics
        # print(current_metrics)
        metrics.append(('env_id', current_metrics))

        if save:
            out_dir = join(ETL_PATH, f'{model_name}/{params}')
            if not exists(out_dir):
                makedirs(out_dir)
            out = join(out_dir, f'{dataset}_{model_name}_{params}_{nbtopics}')
            print('saving to', out)
            df_lda.to_csv(out + '.csv')
            ldamodel.save(out)
            with open(out + '_stats.json', 'w') as fp:
                json.dump(stats, fp)
            with open(out + '_metrics.json', 'w') as fp:
                serializable_metrics = {}
                for k, v in current_metrics.items():
                    if isinstance(v[0], np.ndarray):
                        serializable_metrics[k] = [x.tolist() for x in v]
                    else:
                        serializable_metrics[k] = [float(x) for x in v]
                json.dump(serializable_metrics, fp)
