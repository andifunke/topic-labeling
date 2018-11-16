# coding: utf-8

import argparse
import re
from os.path import join
from time import time

import pandas as pd
from numpy import sqrt, newaxis, dot
from gensim import matutils
from gensim.models import Word2Vec, Doc2Vec

from train_w2v import EpochLogger, EpochSaver
from constants import ETL_PATH, DATASETS
from utils import tprint
from tqdm import tqdm
tqdm.pandas()


def get_labels(topic, nb_labels, d2v, w2v, w2v_indexed, d_indices, w_indices):
    # TODO: simplify
    topic_len = len(topic)

    val_d2v = 0.0
    val_w2v = 0.0
    store_indices = []
    for term in topic:
        try:
            # The word2vec value of topic word from doc2vec trained model
            temp_d2v = d2v.wv.vectors_norm[d2v.wv.vocab[term].index]
        except KeyError:
            pass
        else:
            # Getting the unit vector
            mean_d2v = matutils.unitvec(temp_d2v)
            # The dot product of all labels in doc2vec with the unit vector of topic word
            dists_d2v = dot(d2v.docvecs.vectors_docs_norm, mean_d2v)
            val_d2v += dists_d2v

        try:
            temp_w2v = w2v.wv.vectors_norm[w2v.wv.vocab[term].index]
            # The word2vec value of topic word from word2vec trained model
        except KeyError:
            pass
        else:
            # Unit vector
            mean_w2v = matutils.unitvec(temp_w2v)
            # dot product of all possible labels in word2vec vocab with the unit vector of the topic term
            dists_w2v = dot(w2v_indexed, mean_w2v)
            """
            This next section of code checks if the topic word is also a potential label in trained 
            word2vec model. If that is the case, it is important the dot product of label with that 
            topic word is not taken into account.Hence we make that zero and further down the code 
            also exclude it in taking average of that label over all topic words. 
            """
            if w2v.wv.vocab[term].index in w_indices:
                i_val = w_indices.index(w2v.wv.vocab[term].index)
                store_indices.append(i_val)
                dists_w2v[i_val] = 0.0
            val_w2v += dists_w2v

    # Give the average vector over all topic words
    avg_d2v = val_d2v / topic_len
    avg_w2v = val_w2v / topic_len

    # This modifies the average w2v vector for cases in which the w2v label was same as topic term.
    for element in store_indices:
        avg_w2v[element] = (avg_w2v[element] * topic_len) / (topic_len - 1)

    # argsort and get top 100 doc2vec label indices
    best_d2v = matutils.argsort(avg_d2v, topn=100, reverse=True)
    best_w2v = matutils.argsort(avg_w2v, topn=100, reverse=True)

    result_d2v = []
    # Get the doc2vec labels from indices
    for element in best_d2v:
        ind = d_indices[element]
        temp = d2v.docvecs.index_to_doctag(ind)
        result_d2v.append((temp, float(avg_d2v[element])))

    # Get the word2vec labels from indices
    result_w2v = []
    for element in best_w2v:
        ind = w_indices[element]
        temp = w2v.wv.index2word[ind]
        result_w2v.append((temp, float(avg_w2v[element])))

    # Get the combined set of both doc2vec labels and word2vec labels
    comb_labels = list(set([j[0] for j in result_d2v] + [k[0] for k in result_w2v]))

    # Get indices from combined labels
    newlist_d2v = []
    newlist_w2v = []
    for word in comb_labels:
        try:
            newlist_d2v.append(d_indices.index(d2v.docvecs.doctags[word].offset))
            newlist_w2v.append(w_indices.index(w2v.wv.vocab[word].index))
        except:
            pass
    newlist_d2v = list(set(newlist_d2v))
    newlist_w2v = list(set(newlist_w2v))

    # Finally again get the labels from indices. We searched for the score from both d2v and w2v models.
    resultlist_d2v_new = [
        (d2v.docvecs.index_to_doctag(d_indices[elem_]), float(avg_d2v[elem_]))
        for elem_ in newlist_d2v
    ]
    resultlist_w2v_new = [
        (w2v.wv.index2word[w_indices[elem_]], float(avg_w2v[elem_]))
        for elem_ in newlist_w2v
    ]

    # Finally get the combined score with the label. The label used will be of doc2vec not of word2vec.
    new_score = []
    for term in resultlist_w2v_new:
        k, v = term
        for elem in resultlist_d2v_new:
            k2, v2 = elem
            if k == k2:
                v3 = v + v2
                new_score.append((k2, v3))
    new_score = sorted(new_score, key=lambda x: x[1], reverse=True)[:nb_labels]
    return new_score


def load_embeddings(d2v_path, w2v_path):
    print("Doc2Vec loading", d2v_path)
    d2v = Doc2Vec.load(d2v_path)
    print('vocab size:', len(d2v.wv.vocab))
    print('docvecs size:', len(d2v.docvecs.vectors_docs))

    print("Word2Vec loading", w2v_path)
    w2v = Word2Vec.load(w2v_path)
    print('vocab size:', len(w2v.wv.vocab))

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


def get_indices(d2v, w2v, max_title_length=4, min_doc_length=40):
    phrases = get_phrases(max_title_length=max_title_length, min_doc_length=min_doc_length)
    d_indices = []
    w_indices = []
    dout = wout = 0
    for label in phrases:
        try:
            idx = d2v.docvecs.doctags[label].offset
            d_indices.append(idx)
        except:
            dout += 1
        try:
            idx = w2v.wv.vocab[label].index
            w_indices.append(idx)
        except:
            wout += 1
    return d_indices, w_indices


def index_embeddings(d2v, w2v, d_indices, w_indices):
    """
    Modifies the argument models. Normalizes the d2v und w2v vectors.
    Also reduces the number of d2v docvecs.
    """
    d_indices = list(set(d_indices))
    w_indices = list(set(w_indices))
    print('d2v indices', len(d_indices))
    print('w2v indices', len(w_indices))

    # Models normalised in unit vectord from the indices given above in pickle files.
    d2v.wv.vectors_norm = (
        d2v.wv.vectors / sqrt((d2v.wv.vectors ** 2).sum(-1))[..., newaxis]
    )
    d2v.docvecs.vectors_docs_norm = (
        d2v.docvecs.vectors_docs / sqrt((d2v.docvecs.vectors_docs ** 2).sum(-1))[..., newaxis]
    )[d_indices]
    print('d2v.wv.vectors_norm', len(d2v.wv.vectors_norm))
    print('d2v.docvecs.vectors_docs_norm', len(d2v.docvecs.vectors_docs_norm))
    print("doc2vec normalized")

    w2v.wv.vectors_norm = (
        w2v.wv.vectors / sqrt((w2v.wv.vectors ** 2).sum(-1))[..., newaxis]
    )
    w2v_indexed = w2v.wv.vectors_norm[w_indices]
    print('w2v.wv.vectors_norm', len(w2v.wv.vectors_norm))
    print('w2v_indexed', len(w2v_indexed))
    print("word2vec normalized")
    return w2v_indexed


def load_topics(topics_path, metrics, params, nbtopics, print_sample=False):
    print("Loading topics", topics_path)
    topics = (
        pd
        .read_csv(topics_path, index_col=[0, 1, 2, 3, 4])
        .query('metric in @metrics and param_id in @params and nb_topics in @nbtopics')
        .reset_index(drop=True)
    )
    if print_sample:
        tprint(topics)
    else:
        print('number of topics', len(topics))
    return topics


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--nbfiles", type=int, required=False, default=None)
    parser.add_argument("--version", type=str, required=False, default='noun')
    parser.add_argument("--topics_file", type=str, required=False)
    parser.add_argument("--labels_file", type=str, required=False)

    emb_path = join(ETL_PATH, 'embeddings')
    parser.add_argument("--d2v_path", type=str, required=False, default=join(emb_path, 'd2v', 'd2v'))
    parser.add_argument("--w2v_path", type=str, required=False, default=join(emb_path, 'w2v', 'w2v'))

    parser.add_argument("--metrics", nargs='*', type=str, required=False, default=['ref'])
    parser.add_argument("--params", nargs='*', type=str, required=False, default=['e42'])
    parser.add_argument("--nbtopics", nargs='*', type=int, required=False, default=[100])
    parser.add_argument("--max_title_length", type=int, required=False, default=4)
    parser.add_argument("--min_doc_length", type=int, required=False, default=40)
    parser.add_argument("--nblabels", type=int, required=False, default=20)

    args = parser.parse_args()

    dataset = DATASETS.get(args.dataset, args.dataset)
    nbfiles_str = f'_{args.nbfiles:02d}' if args.nbfiles else ''
    if args.topics_file is None:
        topics_file = join(
            ETL_PATH, 'LDAmodel', args.version, 'Reranker',
            f'{dataset}{nbfiles_str}_topic-candidates.csv'
        )
    else:
        topics_file = args.topics_file
    if args.labels_file is None:
        labels_file = join(
            ETL_PATH, 'LDAmodel', args.version, 'Reranker',
            f'{dataset}{nbfiles_str}_label-candidates.csv'
        )
    else:
        labels_file = args.topics_file

    return (
        topics_file, labels_file, args.d2v_path, args.w2v_path,
        args.metrics, args.params, args.nbtopics,
        args.max_title_length, args.min_doc_length, args.nblabels
    )


def main():
    (
        topics_file, labels_file, d2v_path, w2v_path,
        metrics, params, nb_topics,
        max_title_length, min_doc_length, nb_labels
    ) = parse_args()

    topics = load_topics(topics_file, metrics, params, nb_topics, print_sample=True)
    d2v, w2v = load_embeddings(d2v_path, w2v_path)
    d_indices, w_indices = get_indices(d2v, w2v, max_title_length, min_doc_length)
    w2v_indexed = index_embeddings(d2v, w2v, d_indices, w_indices)

    t0 = time()
    labels = topics.progress_apply(
        lambda row: get_labels(
            topic=row, nb_labels=nb_labels, d2v=d2v, w2v=w2v, w2v_indexed=w2v_indexed,
            d_indices=d_indices, w_indices=w_indices
        ),
        axis=1
    )
    tprint(labels)
    print("Writing labels to", labels_file)
    labels.to_csv(labels_file)
    t1 = int(time() - t0)
    print(f"done in {t1//3600:02d}:{(t1//60) % 60:02d}:{t1 % 60:02d}")


if __name__ == '__main__':
    main()
