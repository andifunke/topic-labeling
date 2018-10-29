# coding: utf-8

from os import listdir, makedirs
from os.path import join, isfile, isdir, exists
import pandas as pd
import gc
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel, LdaModel, LdaMulticore
from gensim.models import CoherenceModel
from gensim.models.callbacks import CoherenceMetric, DiffMetric, PerplexityMetric, ConvergenceMetric

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from constants import FULL_PATH, ETL_PATH, NLP_PATH, SMPL_PATH, POS, NOUN, PROPN, TOKEN, HASH, \
    SENT_IDX, PUNCT

pd.options.display.max_rows = 2001


def docs_to_lists(token_series):
    return tuple(token_series.tolist())


def docs2corpus(documents, tfidf=True, stopwords=None, filter_below=5, filter_above=0.5):
    dictionary = Dictionary(documents)
    dictionary.filter_extremes(no_below=filter_below, no_above=filter_above)

    # filter some noice (e.g. special characters)
    if stopwords:
        stopword_ids = [dictionary.token2id[token] for token in stopwords]
        dictionary.filter_tokens(bad_ids=stopword_ids, good_ids=None)

    bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
    if tfidf:
        tfidf_model = TfidfModel(bow_corpus)
        tfidf_corpus = tfidf_model[bow_corpus]
        return tfidf_corpus, dictionary
    else:
        return bow_model, dictionary


def get_model(corpus, dictionary, nbtopics=100, save=False, parametrization='x42'):

    # Choose α from [0.05, 0.1, 0.5, 1, 5, 10]
    # Choose β from [0.05, 0.1, 0.5, 1, 5, 10]

    print(f'building LDA model "{parametrization}" with {nbtopics} number of topics')
    # default model name: x42_...
    ldamodels = {
        'x42': dict(
            random_state=42,
            corpus=tfidf_corpus,
            alpha='auto',
            eta='auto',
            id2word=dictionary,
            num_topics=nbtopics,
            chunksize=2000,
            passes=20,
            eval_every=10,
            iterations=1000,
        ),
        'a42': dict(
            random_state=42,
            corpus=tfidf_corpus,
            id2word=dictionary,
            num_topics=nbtopics,
        ),
        'b42': dict(
            random_state=42,
            corpus=tfidf_corpus,
            id2word=dictionary,
            num_topics=nbtopics,
            passes=10, iterations=100,
        ),
        'c42': dict(
            random_state=42,
            corpus=tfidf_corpus,
            id2word=dictionary,
            num_topics=nbtopics,
            passes=10, iterations=10_000,
        ),
        'd42': dict(
            random_state=42,
            corpus=tfidf_corpus,
            id2word=dictionary,
            num_topics=nbtopics,
            passes=10, iterations=200,
            eta=0.01,
            alpha=0.1,
        ),
    }
    ldamodel = LdaModel(**ldamodels[parametrization])

    topics = [
        [dataset] + [dictionary[term[0]]
                     for term in ldamodel.get_topic_terms(i)] for i in range(nbtopics)
    ]
    df_topics = pd.DataFrame(topics, columns=['dataset'] + ['term' + str(i) for i in range(10)])

    if save:
        out_dir = join(ETL_PATH, f'LDAmodel/{parametrization}')
        if not exists(out_dir):
            makedirs(out_dir)

        out = join(out_dir, f'{dataset}_ldamdodel_{parametrization}_{nbtopics}')
        print('saving to', out)
        df_topics.to_csv(out + '.csv')
        ldamodel.save(out)


datasets = [
    'Europarl',
    'FAZ_combined',
    'FOCUS_cleansed',
    'OnlineParticipation',
    'PoliticalSpeeches',
    'dewiki',
    'dewac',
]

goodids = {
    # filetered via some fixed rules and similarity measure to character distribution
    'dewac': join(ETL_PATH, 'dewac_good_ids.pickle'),
    'dewiki': join(ETL_PATH, 'dewiki_good_ids.pickle'),

    # the samples contain only a small subset of all articles
    # the reason for this is that the samples are roughly equal in size per category
    'FAZ_combined': join(ETL_PATH, 'FAZ_document_sample3.pickle'),
    'FOCUS_cleansed': join(ETL_PATH, 'FOCUS_document_sample3.pickle'),
}

bad_tokens = {
    'Europarl': [
        'o', 'E.', 'Kerr', 'The', 'la', 'ia', 'For', 'Ieke', 'the',
    ],
    'FAZ_combined': [
        'S.', 'G', 'j.reinecke@faz.de', 'Z', 'B.', 'P', 'E',
    ],
    'FOCUS_cleansed': [],
    'OnlineParticipation': [
        'Re', '@#1', '@#2', '@#3', '@#4', '@#5', '@#6', '@#7', '@#8', '@#9', '@#1.1', 'Für', 'Muss',
        'etc', 'sorry', 'Ggf', 'u.a.',
        'B.', 'stimmt', ';-)', 'lieber', 'o.', 'Ja', 'Desweiteren',
    ],
    'PoliticalSpeeches': [],
    'dewiki': [],
    'dewac': [],
}

for dataset in datasets[1:2]:
    print('dataset:', dataset)
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

    documents = []
    for name in files:
        gc.collect()
        full_path = join(dir_path, name)
        if not isfile(full_path):
            continue

        print('read', name)
        df = pd.read_pickle(join(dir_path, name))
        print('size:', len(df))
        if keepids is not None:
            # some datasets have already been filtered so you may not see a difference in any case
            df = df[df.hash.isin(keepids.index)]
            print('keep:', len(df))

        # fixing bad POS tagging
        mask = df.token.isin(['[', ']', '<', '>', '/', '–', '%'])
        df.loc[mask, POS] = PUNCT

        # using only certain POS tags
        df = df[df[POS].isin({NOUN, PROPN, 'NER', 'NPHRASE'})]
        df = df.groupby([HASH])[TOKEN].agg(docs_to_lists)
        documents += df.values.tolist()

    del keepids, files
    gc.collect()

    params = ['x42', 'a42', 'b42', 'c42', 'd42'][-1]
    if documents:
        corpus, dic = docs2corpus(documents, tfidf=True, stopwords=None, filter_below=5, filter_above=0.5)

        ldamodeling(
            documents,
            nbtopics=50,
            stopwords=bad_tokens[dataset],
            save=True,
            parametrization=params,
        )
