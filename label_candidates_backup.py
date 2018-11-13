"""
Author:         Shraey Bhatia
Date:           October 2016
File:           candidate_gen.py

This file generates label candidates and save the output in a file. It uses both
doc2vec and word2vec models and normalise them to unit vector. There are a couple of
pickle files namely doc2vec_indices and word2vec_indices  which restrict the search of
word2vec and doc2vec labels. These pickle files are in support_files.
"""
from functools import partial
from os.path import join

import pandas as pd
from gensim.models import Word2Vec, Doc2Vec
from numpy import float32, sqrt, newaxis, dot
from gensim import matutils
import argparse

from constants import ETL_PATH
from train_w2v import EpochLogger, EpochSaver
from utils import tprint
import multiprocessing as mp


def get_labels(topic_kv, d2v, w2v, w2v_indexed, d_indices, w_indices):
    # TODO: simplify
    print(topic_kv)
    topic_num, topic = topic_kv
    topic_len = len(topic)
    print("Processing Topic number", topic_num)
    print(topic)

    val_d2v = 0.0
    val_w2v = 0.0
    store_indices = []
    for term in topic:
        try:
            # The word2vec value of topic word from doc2vec trained model
            temp_d2v = d2v.wv.vectors_norm[d2v.wv.vocab[term].index]
            # print(term, 'in d2v.wv')
        except KeyError:
            pass
        else:
            # Getting the unit vector
            mean_d2v = matutils.unitvec(temp_d2v).astype(float32)
            # The dot product of all labels in doc2vec with the unit vector of topic word
            dists_d2v = dot(d2v.docvecs.vectors_docs_norm, mean_d2v)
            val_d2v += dists_d2v

        try:
            temp_w2v = w2v.wv.vectors_norm[w2v.wv.vocab[term].index]
            # The word2vec value of topic word from word2vec trained model
            # print(term, 'in w2v.wv')
        except KeyError:
            pass
        else:
            # Unit vector
            mean_w2v = matutils.unitvec(temp_w2v).astype(float32)
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
    new_score = sorted(new_score, key=lambda x: x[1], reverse=True)

    print(new_score)
    return new_score


def load_embeddings(d2v_path, w2v_path):
    print("Doc2Vec loading", d2v_path)
    d2v = Doc2Vec.load(d2v_path)
    print('vocab size:', len(d2v.wv.vocab))
    print('docvecs size:', len(d2v.docvecs.vectors_docs))
    print("Word2Vec loading", w2v_path)
    w2v = Word2Vec.load(w2v_path)
    print('vocab size:', len(w2v.wv.vocab))
    print("models loaded")
    return d2v, w2v


def get_phrases(max_title_length, min_doc_length):
    dewiki_phrases_lemmatized = 'dewiki_phrases_lemmatized.pickle'
    dpl = pd.read_pickle(join(ETL_PATH, dewiki_phrases_lemmatized))
    # creating a list containing original and lemmatized phrases
    print(max_title_length, min_doc_length)
    print(len(dpl))
    dpl = dpl.query(f"doc_len >= {min_doc_length} and title_len <= {max_title_length}")
    print(len(dpl))
    phrases = dpl.text.append(dpl.token).tolist()
    print('phrases len', len(phrases))
    phrases = set(phrases)
    print('phrases len', len(phrases))
    print(sorted(phrases)[:100])
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
        (
                d2v.wv.vectors /
                sqrt((d2v.wv.vectors ** 2).sum(-1))[..., newaxis]
        )
        .astype(float32)
    )
    d2v.docvecs.vectors_docs_norm = (
        (
                d2v.docvecs.vectors_docs /
                sqrt((d2v.docvecs.vectors_docs ** 2).sum(-1))[..., newaxis]
        )
        .astype(float32)[d_indices]
    )
    print('d2v.wv.vectors_norm', len(d2v.wv.vectors_norm))
    print('d2v.docvecs.vectors_docs_norm', len(d2v.docvecs.vectors_docs_norm))
    print("doc2vec normalized")

    w2v.wv.vectors_norm = (
        (
                w2v.wv.vectors /
                sqrt((w2v.wv.vectors ** 2).sum(-1))[..., newaxis]
        )
        .astype(float32)
    )
    w2v_indexed = w2v.wv.vectors_norm[w_indices]
    print('w2v.wv.vectors_norm', len(w2v.wv.vectors_norm))
    print('w2v_indexed', len(w2v_indexed))
    print("word2vec normalized")
    return w2v_indexed


def load_topics(topics_path, print_sample=False):
    # Loading the data file
    print("Loading topics", topics_path)
    topics = pd.read_csv(topics_path)
    if print_sample:
        tprint(topics, 10)
    try:
        new_frame = topics.drop('domain', 1)
        topic_list = new_frame.set_index('topic_id').T.to_dict('list')
    except:
        topic_list = topics.set_index('topic_id').T.to_dict('list')
    print('number of topics', len(topic_list))
    print(topic_list)
    return topic_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--topics_file", type=str, required=False, default=join(ETL_PATH))
    parser.add_argument("--labels_file", type=str, required=True)

    emb_path = join(ETL_PATH, 'embeddings')
    parser.add_argument("--d2v_path", type=str, required=False, default=join(emb_path, 'd2v', 'd2v'))
    parser.add_argument("--w2v_path", type=str, required=False, default=join(emb_path, 'w2v', 'w2v'))

    parser.add_argument('--cacheinmem', dest='cache_in_memory', action='store_true', required=False)
    parser.add_argument('--no-cacheinmem', dest='cache_in_memory', action='store_false', required=False)
    parser.set_defaults(cache_in_memory=False)

    parser.add_argument("--max_title_length", type=int, required=False, default=4)
    parser.add_argument("--min_doc_length", type=int, required=False, default=40)
    parser.add_argument("--cores", type=int, required=False, default=1)

    args = parser.parse_args()
    return (
        join(ETL_PATH, args.topics_file), join(ETL_PATH, args.labels_file),
        args.d2v_path, args.w2v_path,
        args.max_title_length, args.min_doc_length, args.cores
    )


def main():
    topics_file, labels_file, d2v_path, w2v_path, max_title_length, min_doc_length, cores = parse_args()

    d2v, w2v = load_embeddings(d2v_path, w2v_path)
    d_indices, w_indices = get_indices(d2v, w2v, max_title_length, min_doc_length)
    w2v_indexed = index_embeddings(d2v, w2v, d_indices, w_indices)
    topics_list = load_topics(topics_file)

    labels = partial(
        get_labels,
        d2v=d2v, w2v=w2v, w2v_indexed=w2v_indexed, d_indices=d_indices, w_indices=w_indices
    )

    # *** Error in `/home/andreas/bin/anaconda3/envs/gensim2/bin/python':
    # corrupted double-linked list (not small): 0x0000559234a9d0a0 ***
    # > try single-threaded
    if cores > 1:
        pool = mp.Pool(processes=cores)
        result = pool.map(labels, topics_list.items())
    else:
        result = map(labels, topics_list.items())

    # Write output candidates
    # TODO: convert to pandas
    with open(labels_file, 'w') as fp:
        for i, elem in enumerate(result):
            val = ""
            for item in elem:
                val = val + " " + item[0]
            fp.write(val + "\n")

    print("Candidate labels written to", labels_file)
    print("\n")


if __name__ == '__main__':
    main()
