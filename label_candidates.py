# coding: utf-8

import argparse
import pickle
import re
from os.path import join
from time import time

import pandas as pd
from numpy import dot, float32 as REAL, sqrt, newaxis
from gensim import matutils
from gensim.models import Word2Vec, Doc2Vec

from train_utils import init_logging, log_args
from train_w2v import EpochLogger, EpochSaver
from constants import ETL_PATH, DATASETS
LOG = None


def get_word(word):
    if type(word) != str:
        return word
    inst = re.search(r"_\(([A-Za-z0-9_]+)\)", word)
    if inst is None:
        return word
    else:
        word = re.sub(r'_\(.+\)', '', word)
        return word


def get_labels(topic, nb_labels, d2v, w2v, w2v_indexed, d_indices, w_indices):
    valdoc2vec = 0.0
    valword2vec = 0.0
    store_indices = []
    topic_len = len(topic)
    for item in topic:
        try:
            # The word2vec value of topic word from doc2vec trained model
            tempdoc2vec = d2v.wv.syn0norm[d2v.wv.vocab[item].index]
        except:
            pass
        else:
            meandoc2vec = matutils.unitvec(tempdoc2vec).astype(REAL)  # Getting the unit vector
            # The dot product of all labels in doc2vec with the unit vector of topic word
            distsdoc2vec = dot(d2v.docvecs.doctag_syn0norm, meandoc2vec)
            valdoc2vec = valdoc2vec + distsdoc2vec

        try:
            # The word2vec value of topic word from word2vec trained model
            tempword2vec = w2v.wv.syn0norm[w2v.wv.vocab[item].index]
        except:
            pass
        else:
            # Unit vector
            meanword2vec = matutils.unitvec(tempword2vec).astype(REAL)
            # dot product of all possible labels in word2vec vocab with the unit vector of topic word
            distsword2vec = dot(w2v_indexed, meanword2vec)
            """
            This next section of code checks if the topic word is also a potential label in trained 
            word2vec model. If that is the case, it is important the dot product of label with that 
            topic word is not taken into account.Hence we make that zero and further down the code 
            also exclude it in taking average of that label over all topic words. 
            """
            if w2v.wv.vocab[item].index in w_indices:
                i_val = w_indices.index(w2v.wv.vocab[item].index)
                store_indices.append(i_val)
                distsword2vec[i_val] = 0.0
            valword2vec = valword2vec + distsword2vec

    avgdoc2vec = valdoc2vec / float(topic_len)  # Give the average vector over all topic words
    avgword2vec = valword2vec / float(topic_len)  # Average of word2vec vector over all topic words

    # argsort and get top 100 doc2vec label indices
    bestdoc2vec = matutils.argsort(avgdoc2vec, topn=100, reverse=True)
    resultdoc2vec = []
    # Get the doc2vec labels from indices
    for elem in bestdoc2vec:
        ind = d_indices[elem]
        temp = d2v.docvecs.index_to_doctag(ind)
        resultdoc2vec.append((temp, float(avgdoc2vec[elem])))

    # This modifies the average word2vec vector for cases
    # in which the word2vec label was same as topic word.
    for element in store_indices:
        avgword2vec[element] = (avgword2vec[element] * topic_len) / (float(topic_len - 1))

    # argsort and get top 100 word2vec label indices
    bestword2vec = matutils.argsort(avgword2vec, topn=100, reverse=True)
    # Get the word2vec labels from indices
    resultword2vec = []
    for element in bestword2vec:
        ind = w_indices[element]
        temp = w2v.wv.index2word[ind]
        resultword2vec.append((temp, float(avgword2vec[element])))

    # Get the combined set of both doc2vec labels and word2vec labels
    comb_labels = sorted(set([i[0] for i in resultdoc2vec] + [i[0] for i in resultword2vec]))
    newlist_doc2vec = []
    newlist_word2vec = []

    # Get indices from combined labels
    for elem in comb_labels:
        try:
            newlist_doc2vec.append(d_indices.index(d2v.docvecs.doctags[elem].offset))
            temp = get_word(elem)
            newlist_word2vec.append(w_indices.index(w2v.wv.vocab[temp].index))
        except:
            pass
    newlist_doc2vec = sorted(set(newlist_doc2vec))
    newlist_word2vec = sorted(set(newlist_word2vec))

    # Finally again get the labels from indices. We searched for the score from both d2v and w2v models
    resultlist_doc2vecnew = [
        (d2v.docvecs.index_to_doctag(d_indices[elem]), float(avgdoc2vec[elem]))
        for elem in newlist_doc2vec
    ]
    resultlist_word2vecnew = [
        (w2v.wv.index2word[w_indices[elem]], float(avgword2vec[elem]))
        for elem in newlist_word2vec
    ]

    # Finally get the combined score with the label. The label used will be of doc2vec not of word2vec.
    new_score = []
    for item in resultlist_word2vecnew:
        k, v = item
        for elem in resultlist_doc2vecnew:
            k2, v2 = elem
            k3 = get_word(k2)
            if k == k3:
                v3 = v + v2
                new_score.append((k2, v3))

    resultlist_doc2vecnew = sorted(resultlist_doc2vecnew, key=lambda x: x[1], reverse=True)[:nb_labels]
    resultlist_word2vecnew = sorted(resultlist_word2vecnew, key=lambda x: x[1], reverse=True)[:nb_labels]
    new_score = sorted(new_score, key=lambda x: x[1], reverse=True)[:nb_labels]
    return [resultlist_doc2vecnew, resultlist_word2vecnew, new_score]


def load_embeddings(d2v_path, w2v_path):
    LOG.info(f'Doc2Vec loading {d2v_path}')
    d2v = Doc2Vec.load(d2v_path)
    LOG.info(f'vocab size: len(d2v.wv.vocab)')
    LOG.info(f'docvecs size: {len(d2v.docvecs.vectors_docs)}')

    LOG.info(f'Word2Vec loading {w2v_path}')
    w2v = Word2Vec.load(w2v_path)
    LOG.info(f'vocab size: {len(w2v.wv.vocab)}')
    return d2v, w2v


def get_phrases(max_title_length, min_doc_length, lemmatized_only=True):
    dewiki_phrases_lemmatized = 'dewiki_phrases_lemmatized.pickle'
    phrases = pd.read_pickle(join(ETL_PATH, dewiki_phrases_lemmatized))
    # creating a list containing original and lemmatized phrases
    phrases = phrases.query(f"doc_len >= {min_doc_length} and title_len <= {max_title_length}")
    if lemmatized_only:
        phrases = phrases.token.unique()
    else:
        phrases = phrases.token.append(phrases.text).unique()
    pat = re.compile(r'^[a-zA-ZÄÖÜäöü]+.*')
    phrases = filter(lambda x: pat.match(x), phrases)
    return phrases


def get_indices(d2v, w2v, max_title_length=4, min_doc_length=41):
    phrases = get_phrases(max_title_length=max_title_length, min_doc_length=min_doc_length)
    d2v_indices = []
    w2v_indices = []
    dout = wout = 0
    for label in phrases:
        try:
            idx = d2v.docvecs.doctags[label].offset
            d2v_indices.append(idx)
        except:
            dout += 1
        try:
            idx = w2v.wv.vocab[label].index
            w2v_indices.append(idx)
        except:
            wout += 1
    return d2v_indices, w2v_indices


def index_embeddings(d2v, w2v, d2v_indices, w2v_indices):
    """
    Modifies the argument models. Normalizes the d2v und w2v vectors.
    Also reduces the number of d2v docvecs.
    """
    # Models normalised in unit vectord from the indices given above in pickle files.
    d2v.wv.syn0norm = (d2v.wv.syn0 / sqrt((d2v.wv.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)
    d2v.docvecs.vectors_docs_norm = \
        (d2v.docvecs.doctag_syn0 / sqrt((d2v.docvecs.doctag_syn0 ** 2).sum(-1))[..., newaxis]) \
        .astype(REAL)[d2v_indices]
    LOG.info("doc2vec normalized")

    w2v.wv.syn0norm = (w2v.wv.syn0 / sqrt((w2v.wv.syn0 ** 2).sum(-1))[..., newaxis]).astype(
        REAL)
    w2v_indexed = w2v.wv.syn0norm[w2v_indices]
    LOG.info("word2vec normalized")
    return w2v_indexed


def load_topics(topics_path, metrics, params, nbtopics, print_sample=False):
    LOG.info(f'Loading topics {topics_path}')
    topics = pd.read_csv(topics_path, index_col=None)
    if metrics and 'metric' in topics.columns:
        topics = topics[topics.metric.isin(metrics)]
    if params and 'param_id' in topics.columns:
        topics = topics[topics.param_id.isin(params)]
    if nbtopics and 'nb_topics' in topics.columns:
        topics = topics[topics.nb_topics.isin(nbtopics)]
    topics = (
        topics.drop(
            ['dataset', 'metric', 'param_id', 'nb_topics', 'topic_idx', 'topic_id', 'domain'],
            axis=1, errors='ignore'
        )
        .reset_index(drop=True)
    )
    if print_sample:
        LOG.info('\n' + topics.head(10))
    LOG.info(f'number of topics {len(topics)}')
    return topics


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--topics_file", type=str, required=False)
    parser.add_argument("--labels_file", type=str, required=False)
    parser.add_argument("--d2v_indices", type=str, required=False)
    parser.add_argument("--w2v_indices", type=str, required=False)

    emb_path = join(ETL_PATH, 'embeddings')
    parser.add_argument("--d2v_path", type=str, required=False, default=join(emb_path, 'd2v', 'd2v'))
    parser.add_argument("--w2v_path", type=str, required=False, default=join(emb_path, 'w2v', 'w2v'))

    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--nbfiles", type=int, required=False, default=None)
    parser.add_argument("--version", type=str, required=False, default='noun')
    parser.add_argument("--metrics", nargs='*', type=str, required=False, default=['ref'])
    parser.add_argument("--params", nargs='*', type=str, required=False, default=['e42'])
    parser.add_argument("--nbtopics", nargs='*', type=int, required=False, default=[100])

    parser.add_argument("--nblabels", type=int, required=False, default=20)
    parser.add_argument("--max_title_length", type=int, required=False, default=4)
    parser.add_argument("--min_doc_length", type=int, required=False, default=41)

    args = parser.parse_args()

    dataset = DATASETS.get(args.dataset, args.dataset)
    nbfiles_str = f'_nbfiles{args.nbfiles:02d}' if args.nbfiles else ''
    if args.topics_file:
        topics_file_is_given = True
        args.nbfiles = args.version = args.metrics = args.params = args.nbtopics = None
    else:
        topics_file_is_given = False
        args.topics_file = join(
            ETL_PATH, 'LDAmodel', args.version, 'Reranker',
            f'{dataset}{nbfiles_str}_topic-candidates.csv'
        )
    if args.labels_file is None:
        if topics_file_is_given:
            args.labels_file = args.topics_file.strip('.csv') + '_label-candidates.csv'
        else:
            args.labels_file = join(
                ETL_PATH, 'LDAmodel', args.version, 'Reranker',
                f'{dataset}{nbfiles_str}_label-candidates.csv'
            )

    if args.d2v_indices and args.w2v_indices:
        args.max_title_length = None
        args.min_doc_length = None

    return (
        args.topics_file, args.labels_file, args.d2v_indices, args.w2v_indices,
        args.d2v_path, args.w2v_path,
        args.metrics, args.params, args.nbtopics,
        args.max_title_length, args.min_doc_length, args.nblabels, args
    )


def main():
    global LOG
    (
        topics_file, labels_file, d2v_indices_file, w2v_indices_file,
        d2v_path, w2v_path,
        metrics, params, nb_topics,
        max_title_length, min_doc_length, nb_labels, args
    ) = parse_args()

    LOG = logger = init_logging(name='label_candidates')
    log_args(logger, args)

    topics = load_topics(
        topics_path=topics_file, metrics=metrics, params=params, nbtopics=nb_topics, print_sample=False
    )

    d2v, w2v = load_embeddings(
        d2v_path=d2v_path, w2v_path=w2v_path
    )

    if d2v_indices_file and w2v_indices_file:
        logger.info(f'loading {d2v_indices_file}')
        with open(d2v_indices_file, 'rb') as fp:
            d2v_indices = pickle.load(fp)
        logger.info(f'loading {w2v_indices_file}')
        with open(w2v_indices_file, 'rb') as fp:
            w2v_indices = pickle.load(fp)
    else:
        d2v_indices, w2v_indices = get_indices(
            d2v=d2v, w2v=w2v, max_title_length=max_title_length, min_doc_length=min_doc_length
        )
    d2v_indices = sorted(set(d2v_indices))
    w2v_indices = sorted(set(w2v_indices))

    w2v_indexed = index_embeddings(
        d2v=d2v, w2v=w2v, d2v_indices=d2v_indices, w2v_indices=w2v_indices
    )

    t0 = time()
    labels = topics.apply(
        lambda row: get_labels(
            topic=row, nb_labels=nb_labels,
            d2v=d2v, w2v=w2v, w2v_indexed=w2v_indexed,
            d_indices=d2v_indices, w_indices=w2v_indices
        ),
        axis=1
    )
    t1 = int(time() - t0)
    logger.info(f"done in {t1//3600:02d}:{(t1//60) % 60:02d}:{t1 % 60:02d}")

    # reformatting output files
    full = (
        labels
        .apply(pd.Series)
        .rename(columns={0: 'd2v', 1: 'w2v', 2: 'comb'})
        .stack()
        .apply(pd.Series)
        .rename(columns=lambda x: f'label{x}')
    )
    simple = (
        full[full.index.get_level_values(1) == 'comb']
        .reset_index(drop=True)
        .applymap(lambda x: x[0])
    )

    logger.info(f'Writing labels to {labels_file}')
    full.to_csv(labels_file + '_full.csv')
    simple.to_csv(labels_file + '_simple.csv')


if __name__ == '__main__':
    main()
