"""
Author:         Shraey Bhatia
Date:           October 2016
File:           candidate_gen.py

This file generates label candidates and save the output in a file. It uses both
doc2vec and word2vec models and normalise them to unit vector. There are a couple of
pickle files namely doc2vec_indices and word2vec_indices  which restrict the search of
word2vec and doc2vec labels. These pickle files are in support_files.
"""

import pandas as pd
from gensim.models import Word2Vec, Doc2Vec
from numpy import float32, sqrt, newaxis, dot
from gensim import matutils
import re
import pickle
import argparse
from train_w2v import EpochLogger, EpochSaver
from utils import tprint
# import multiprocessing as mp


d2v_model = None
w2v_model = None
w2v_model_indexed = None
w_indices = None
d_indices = None
args = None


def get_word(word):
    """ This method is mainly used to remove brackets from the candidate labels. """
    if type(word) != str:
        return word
    inst = re.search(r"_\(([A-Za-z0-9_]+)\)", word)
    if inst is None:
        return word
    else:
        # print(word)
        word = re.sub(r'_\(.+\)', '', word)
        # print (">>>", word)
        return word


def get_labels(topic_kv):
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
            temp_d2v = d2v_model.wv.vectors_norm[d2v_model.wv.vocab[term].index]
            # print(term, 'in d2v.wv')
        except KeyError:
            print(term, 'not in d2v.wv')
            pass
        else:
            # Getting the unit vector
            mean_d2v = matutils.unitvec(temp_d2v).astype(float32)
            # The dot product of all labels in doc2vec with the unit vector of topic word
            dists_d2v = dot(d2v_model.docvecs.vectors_docs_norm, mean_d2v)
            val_d2v = val_d2v + dists_d2v

        try:
            temp_w2v = w2v_model.wv.vectors_norm[w2v_model.wv.vocab[term].index]
            # The word2vec value of topic word from word2vec trained model
            # print(term, 'in w2v.wv')
        except KeyError:
            print(term, 'not in w2v.wv')
            pass
        else:
            # Unit vector
            mean_w2v = matutils.unitvec(temp_w2v).astype(float32)
            # dot product of all possible labels in word2vec vocab with the unit vector of the topic term
            dists_w2v = dot(w2v_model_indexed, mean_w2v)

            """
            This next section of code checks if the topic word is also a potential label in trained 
            word2vec model. If that is the case, it is important the dot product of label with that 
            topic word is not taken into account.Hence we make that zero and further down the code 
            also exclude it in taking average of that label over all topic words. 
            """
            if w2v_model.wv.vocab[term].index in w_indices:
                i_val = w_indices.index(w2v_model.wv.vocab[term].index)
                store_indices.append(i_val)
                dists_w2v[i_val] = 0.0

            val_w2v = val_w2v + dists_w2v

    # Give the average vector over all topic words
    avg_d2v = val_d2v / topic_len
    avg_w2v = val_w2v / topic_len

    # This modifies the average w2v vector for cases in which the w2v label was same as topic term.
    for element in store_indices:
        # print('> equal', element)
        avg_w2v[element] = (avg_w2v[element] * topic_len) / (topic_len - 1)
    print(avg_w2v)

    # argsort and get top 100 doc2vec label indices
    best_d2v = matutils.argsort(avg_d2v, topn=100, reverse=True)
    best_w2v = matutils.argsort(avg_w2v, topn=100, reverse=True)
    print(best_w2v)

    result_d2v = []
    # Get the doc2vec labels from indices
    for element in best_d2v:
        ind = d_indices[element]
        temp = d2v_model.docvecs.index_to_doctag(ind)
        result_d2v.append((temp, float(avg_d2v[element])))

    # Get the word2vec labels from indices
    result_w2v = []
    for element in best_w2v:
        # print('* in best_w2v', element)
        ind = w_indices[element]
        temp = w2v_model.wv.index2word[ind]
        result_w2v.append((temp, float(avg_w2v[element])))

    # Get the combined set of both doc2vec labels and word2vec labels
    comb_labels = list(set([j[0] for j in result_d2v] + [k[0] for k in result_w2v]))

    # Get indices from combined labels
    newlist_d2v = []
    newlist_w2v = []
    print(comb_labels)
    for elem in comb_labels:
        print(elem)
        try:
            newlist_d2v.append(d_indices.index(d2v_model.docvecs.doctags[elem].offset))
            temp = get_word(elem)
            print(temp)
            newlist_w2v.append(w_indices.index(w2v_model.wv.vocab[temp].index))
        except:
            print('!> except')
            pass
    newlist_d2v = list(set(newlist_d2v))
    newlist_w2v = list(set(newlist_w2v))
    print('newlist_d2v', newlist_d2v)
    print('newlist_d2v', newlist_d2v)

    # Finally again get the labels from indices. We searched for the score from both d2v and w2v models.
    resultlist_d2v_new = [
        (d2v_model.docvecs.index_to_doctag(d_indices[elem_]), float(avg_d2v[elem_]))
        for elem_ in newlist_d2v
    ]
    resultlist_w2v_new = [
        (w2v_model.wv.index2word[w_indices[elem_]], float(avg_w2v[elem_]))
        for elem_ in newlist_w2v
    ]

    # Finally get the combined score with the label. The label used will be of doc2vec not of word2vec.
    new_score = []
    for term in resultlist_w2v_new:
        k, v = term
        for elem in resultlist_d2v_new:
            k2, v2 = elem
            k3 = get_word(k2)
            if k == k3:
                v3 = v + v2
                new_score.append((k2, v3))
    new_score = sorted(new_score, key=lambda x: x[1], reverse=True)

    print(new_score)
    quit()

    return new_score[:(int(args.num_cand_labels))]


def main():
    global args, d_indices, w_indices, d2v_model, w2v_model, w2v_model_indexed

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("num_cand_labels")
    parser.add_argument("doc2vecmodel")
    parser.add_argument("word2vecmodel")
    parser.add_argument("data")
    parser.add_argument("outputfile_candidates")
    parser.add_argument("doc2vec_indices")
    parser.add_argument("word2vec_indices")
    args = parser.parse_args()
    """
    Pickle file needed to run the code. These file have the indices of doc2vec which 
    have length of label(wiki title less than 5). The word2vec file has indices of the
    file which were used in to create phrases in word2vec model. The indices are taken from 
    trained doc2vec and word2vec models. Additionally there is some bit of preprocessing involved 
    of removing brackets from some candidate labels. To get more insight into it refer to the paper.
    """

    with open(args.doc2vec_indices, 'rb') as m:
        print('Loading', args.doc2vec_indices)
        d_indices = pickle.load(m)
    with open(args.word2vec_indices, 'rb') as n:
        print('Loading', args.word2vec_indices)
        w_indices = pickle.load(n)

    print(args.doc2vecmodel)

    # Models loaded
    print("Doc2Vec loading", args.doc2vecmodel)
    d2v_model = Doc2Vec.load(args.doc2vecmodel)
    print('vocab size:', len(d2v_model.wv.vocab))
    print('docvecs size:', len(d2v_model.docvecs.vectors_docs))
    print("Word2Vec loading", args.word2vecmodel)
    w2v_model = Word2Vec.load(args.word2vecmodel)
    print('vocab size:', len(w2v_model.wv.vocab))
    print("models loaded")

    # Loading the data file
    print("Loading topics", args.data)
    topics = pd.read_csv(args.data)
    tprint(topics, 10)

    try:
        new_frame = topics.drop('domain', 1)
        topic_list = new_frame.set_index('topic_id').T.to_dict('list')
    except:
        topic_list = topics.set_index('topic_id').T.to_dict('list')
    # print("Data Gathered")
    print(len(topic_list))

    d_indices = list(set(d_indices))
    w_indices = list(set(w_indices))
    print('d2v indices', len(d_indices))
    print('w2v indices', len(w_indices))

    # Models normalised in unit vectord from the indices given above in pickle files.
    d2v_model.wv.vectors_norm = (
        (
                d2v_model.wv.vectors /
                sqrt((d2v_model.wv.vectors ** 2).sum(-1))[..., newaxis]
        )
        .astype(float32)
    )
    d2v_model.docvecs.vectors_docs_norm = (
        (
                d2v_model.docvecs.vectors_docs /
                sqrt((d2v_model.docvecs.vectors_docs ** 2).sum(-1))[..., newaxis]
        )
        .astype(float32)[d_indices]
    )
    print('d2v_model.wv.vectors_norm', len(d2v_model.wv.vectors_norm))
    print('d2v_model.docvecs.vectors_docs_norm', len(d2v_model.docvecs.vectors_docs_norm))
    print("doc2vec normalized")

    w2v_model.wv.vectors_norm = (
        (
                w2v_model.wv.vectors /
                sqrt((w2v_model.wv.vectors ** 2).sum(-1))[..., newaxis]
        )
        .astype(float32)
    )
    w2v_model_indexed = w2v_model.wv.vectors_norm[w_indices]
    print('w2v_model.wv.vectors_norm', len(w2v_model.wv.vectors_norm))
    print('w2v_model_indexed', len(w2v_model_indexed))
    print("word2vec normalized")

    quit()

    # cores = 4
    # pool = mp.Pool(processes=cores)
    # result = pool.map(get_labels, range(0, len(topic_list)))
    # *** Error in `/home/andreas/bin/anaconda3/envs/gensim2/bin/python':
    # corrupted double-linked list (not small): 0x0000559234a9d0a0 ***
    # > trying single-threaded
    print(topic_list)
    result = map(get_labels, topic_list.items())

    # The output file for candidates.
    with open(args.outputfile_candidates, 'w') as fp:
        for i, elem in enumerate(result):
            val = ""
            for item in elem:
                val = val + " " + item[0]
            fp.write(val + "\n")

    print("Candidate labels written to " + args.outputfile_candidates)
    print("\n")


if __name__ == '__main__':
    main()
