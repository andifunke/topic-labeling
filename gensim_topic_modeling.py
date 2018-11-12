from os import listdir, makedirs
from os.path import join, isfile, exists, dirname
import sys
from sys import stdout
import gc
import pandas as pd
import gensim
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel, LdaModel
from gensim.models.callbacks import CoherenceMetric, DiffMetric, PerplexityMetric, ConvergenceMetric
from constants import ETL_PATH, SMPL_PATH, POS, NOUN, PROPN, TOKEN, HASH, PUNCT, BAD_TOKENS, DATASETS
import logging
import json
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--logger", type=str, required=False, default='shell')
parser.add_argument("--params", nargs='*', type=str, required=False,
                    default=['a42', 'b42', 'c42', 'd42', 'e42'])
parser.add_argument("--nbtopics", nargs='*', type=int, required=False,
                    default=[10, 25, 50, 100])
parser.add_argument("--nbfiles", type=int, required=False, default=None)
args = vars(parser.parse_args())
dataset = DATASETS[args['dataset']]
callback_logger = args['logger']

vis = None
if callback_logger == 'visdom':
    import visdom
    vis = visdom.Visdom()

param_args = args['params']
nbtopics_args = args['nbtopics']

save = True

LOG_PATH = './../logs/LDA_training_{}.log'.format(dataset)
# create path if necessary
makedirs(dirname(LOG_PATH), exist_ok=True)
logger = logging.getLogger('LDA_training')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
fh = logging.FileHandler(LOG_PATH)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# stdout logger
ch = logging.StreamHandler(stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info('pandas: ' + pd.__version__)
logger.info('gensim: ' + gensim.__version__)
logger.info('python: ' + sys.version)
logger.info(param_args)


def docs_to_lists(token_series):
    return tuple(token_series.tolist())


def docs2corpora(documents, tfidf=True, stopwords=None, filter_below=5, filter_above=0.5,
                 split=False, max_test_size_rel=0.1, max_test_size_abs=5000):
    logger.info('Building dictionary')
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
            split2 = int(length * (1 - max_test_size_rel))
        else:
            split2 = length - max_test_size_abs
        training_texts = documents[:split2]
        test_texts = documents[split2:]
        logger.info(
            'split dataset. size of: train_set={:d}, test_set={:d}'
            .format(len(training_texts), len(test_texts))
        )
        corpora['training_corpus'] = [dictionary.doc2bow(text) for text in training_texts]
        corpora['test_corpus'] = [dictionary.doc2bow(text) for text in test_texts]
    else:
        training_texts = documents
        corpora['training_corpus'] = [dictionary.doc2bow(text) for text in training_texts]
        corpora['test_corpus'] = None, None

    if tfidf:
        for key, bow_corpus in corpora.items():
            tfidf_model = TfidfModel(bow_corpus)
            corpora[key] = tfidf_model[bow_corpus]
    return corpora, dictionary


def init_callbacks(training_corpus, test_corpus, documents, viz_env=None, title_suffix=''):
    # define perplexity callback for hold_out and test corpus
    pl_test = PerplexityMetric(
        corpus=test_corpus,
        logger=callback_logger, viz_env=viz_env,
        title="Perplexity (test)" + title_suffix
    )
    # define other remaining metrics available
    ch_umass = CoherenceMetric(
        corpus=training_corpus, coherence="u_mass", topn=10,
        logger=callback_logger, viz_env=viz_env,
        title="Coherence (u_mass)" + title_suffix
    )
    ch_cv = CoherenceMetric(
        corpus=training_corpus, texts=documents, coherence="c_v", topn=10,
        logger=callback_logger, viz_env=viz_env,
        title="Coherence (c_v)" + title_suffix
    )
    diff_kl = DiffMetric(
        distance="kullback_leibler",
        logger=callback_logger, viz_env=viz_env,
        title="Diff (kullback_leibler)" + title_suffix
    )
    convergence_kl = ConvergenceMetric(
        distance="jaccard",
        logger=callback_logger, viz_env=viz_env,
        title="Convergence (jaccard)" + title_suffix
    )
    return [pl_test, ch_umass, ch_cv, diff_kl, convergence_kl]


def get_parameterset(corpus, dictionary, callbacks=None, nbtopics=100, parametrization='a42',
                     eval_every=None):
    logger.info(
        'building LDA model "{}" with {} number of topics'
        .format(parametrization, nbtopics)
    )
    default = dict(
        random_state=42, corpus=corpus, id2word=dictionary, num_topics=nbtopics,
        eval_every=eval_every, callbacks=callbacks, chunksize=20000, dtype=np.float64
    )
    ldamodels = {
        'a42': dict(passes=20),
        'b42': dict(passes=20, iterations=200),
        'c42': dict(passes=20, iterations=1000),
        'd42': dict(passes=20, iterations=200, alpha=0.1, eta=0.01),
        'e42': dict(passes=20, iterations=200, alpha='auto', eta='auto'),
    }
    for key, dic in ldamodels.items():
        dic.update(default)
    return ldamodels[parametrization]


def main():
    goodids = {
        # filetered via some fixed rules and similarity measure to character distribution
        'dewac': join(ETL_PATH, 'dewac_good_ids.pickle'),
        'dewiki': join(ETL_PATH, 'dewiki_good_ids.pickle'),
    }

    # for dataset in datasets[1:2]:
    logger.info('dataset: ' + dataset)
    logger.info('-' * 5)

    sub_dir = 'dewiki' if dataset.startswith('dewi') else 'wiki_phrases'
    dir_path = join(SMPL_PATH, sub_dir)
    files = sorted([f for f in listdir(dir_path) if f.startswith(dataset)])

    keepids = None
    if dataset in goodids:
        keepids = pd.read_pickle(goodids[dataset])

    nb_words = 0
    reduction_pos_set = {NOUN, PROPN, 'NER', 'NPHRASE'}

    nbfiles = args['nbfiles']
    if nbfiles is not None:
        logger.info('processing {} files'.format(nbfiles))

    documents = []
    for name in files[:nbfiles]:
        full_path = join(dir_path, name)
        if not isfile(full_path):
            continue

        logger.info('reading ' + name)
        df = pd.read_pickle(join(dir_path, name))
        logger.info('    initial number of words: ' + str(len(df)))
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
        df = df[~df.token.isin(BAD_TOKENS)]
        nb_words += len(df)
        logger.info('    remaining number of words: ' + str(len(df)))

        # groupby sorts the documents by hash-id
        # which is equal to shuffeling the dataset before building the model
        df = df.groupby([HASH])[TOKEN].agg(docs_to_lists)
        logger.info('    number of documents: ' + str(len(df)))
        documents += df.values.tolist()

    nb_docs = len(documents)
    logger.info('-' * 5)
    logger.info('total number of documents: ' + str(nb_docs))
    logger.info('total number of words: ' + str(nb_words))
    stats = dict(dataset=dataset, pos_set=sorted(reduction_pos_set), nb_docs=nb_docs, nb_words=nb_words)
    del keepids, files
    gc.collect()

    # save full dataset as bow/tfidf corpus in Matrix Market format, including dictionary
    use_tfidf = False
    corpus_type = 'tfidf' if use_tfidf else 'bow'
    split = False
    split_type = 'trainset' if split else 'fullset'
    corpora, dictionary = docs2corpora(
        documents, tfidf=use_tfidf,
        filter_below=5, filter_above=0.5,
        split=split,
    )
    corpus = corpora['training_corpus']
    file_name = '{}_{}_nouns_{}'.format(dataset, split_type, corpus_type)
    corpus_path = join(ETL_PATH, 'LDAmodel', file_name + '.mm')
    dict_path = join(ETL_PATH, 'LDAmodel', file_name + '.dict')
    logger.info('saving ' + corpus_path)
    MmCorpus.serialize(corpus_path, corpus)
    logger.info('saving ' + dict_path)
    dictionary.save(dict_path)
    doc_path = join(ETL_PATH, 'LDAmodel', file_name.rstrip(corpus_type) + 'texts.json')
    logger.info('saving ' + doc_path)
    with open(doc_path, 'w') as fp:
        json.dump(documents, fp, ensure_ascii=False)

    # initialize training and test data
    corpora, dictionary = docs2corpora(
        documents, tfidf=False,
        filter_below=5, filter_above=0.5,
        split=True,
    )
    training_corpus = corpora['training_corpus']
    test_corpus = corpora['test_corpus']

    params_list = param_args
    model_name = 'LDAmodel'
    metrics = []
    topn = 20
    # params = params_list[3]
    for params in params_list:
        env_id = f"{dataset}-{model_name}"
        for nbtopics in nbtopics_args:  # range(10, 101, 10):
            # Choose α from [0.05, 0.1, 0.5, 1, 5, 10]
            # Choose β from [0.05, 0.1, 0.5, 1, 5, 10]
            callbacks = init_callbacks(
                documents=documents,
                training_corpus=training_corpus,
                test_corpus=test_corpus,
                viz_env=env_id,
                title_suffix=f", {params}, {nbtopics}tpx"
            )
            kwargs = get_parameterset(
                training_corpus,
                dictionary,
                callbacks=callbacks,
                nbtopics=nbtopics,
                parametrization=params
            )

            logger.info('running ' + model_name)
            ldamodel = LdaModel(**kwargs)

            topics = [
                [dataset] +
                [dictionary[term[0]] for term in ldamodel.get_topic_terms(i, topn=topn)]
                for i in range(nbtopics)
            ]
            df_lda = pd.DataFrame(topics, columns=['dataset'] + ['term' + str(i) for i in range(topn)])

            current_metrics = ldamodel.metrics
            logger.info(current_metrics)
            metrics.append(('env_id', current_metrics))

            if save:
                out_dir = join(ETL_PATH, model_name, params)
                if not exists(out_dir):
                    makedirs(out_dir)
                out = join(out_dir, '{}_{}_{}_{}'.format(dataset, model_name, params, nbtopics))
                logger.info('saving to ' + out)
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
                if callback_logger == 'visdom' and vis is not None:
                    vis.save([env_id])


if __name__ == '__main__':
    main()
