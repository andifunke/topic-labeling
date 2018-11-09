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
import multiprocessing as mp
import argparse

# Arguments
from utils import tprint

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
model1 = Doc2Vec.load(args.doc2vecmodel)
print("Word2Vec loading", args.word2vecmodel)
model2 = Word2Vec.load(args.word2vecmodel)
print("models loaded")

# Loading the data file
print("Loading topics", args.data)
topics = pd.read_csv(args.data)
# tprint(topics)
try:
    new_frame = topics.drop('domain', 1)
    topic_list = new_frame.set_index('topic_id').T.to_dict('list')
except:
    topic_list = topics.set_index('topic_id').T.to_dict('list')
# print("Data Gathered")
print(len(topic_list))

w_indices = list(set(w_indices))
d_indices = list(set(d_indices))
# print('w2v indices', w_indices[:100])
# print('d2v indices', d_indices[:100])

# Models normalised in unit vectord from the indices given above in pickle files.
model1.syn0norm = (model1.syn0 / sqrt((model1.syn0 ** 2).sum(-1))[..., newaxis]).astype(float32)
model1.docvecs.doctag_syn0norm = (
    (model1.docvecs.doctag_syn0 / sqrt((model1.docvecs.doctag_syn0 ** 2).sum(-1))[..., newaxis])
    .astype(float32)[d_indices]
)
print("doc2vec normalized")

model2.syn0norm = (model2.syn0 / sqrt((model2.syn0 ** 2).sum(-1))[..., newaxis]).astype(float32)
model3 = model2.syn0norm[w_indices]
print("word2vec normalized")

# print(model2.syn0norm[w_indices[:5]])


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


def get_labels(topic_num):
    valdoc2vec = 0.0
    valword2vec = 0.0
    store_indices = []

    print("Processing Topic number " + str(topic_num))
    for item_ in topic_list[topic_num]:
        try:
            tempdoc2vec = model1.syn0norm[
                model1.vocab[item_].index]  # The word2vec value of topic word from doc2vec trained model
        except:
            pass
        else:
            meandoc2vec = matutils.unitvec(tempdoc2vec).astype(float32)  # Getting the unit vector
            distsdoc2vec = dot(model1.docvecs.doctag_syn0norm, meandoc2vec)
            # The dot product of all labels in doc2vec with the unit vector of topic word
            valdoc2vec = valdoc2vec + distsdoc2vec

        try:
            tempword2vec = model2.syn0norm[model2.vocab[item_].index]
            # The word2vec value of topic word from word2vec trained model
        except:
            pass
        else:
            meanword2vec = matutils.unitvec(tempword2vec).astype(float32)  # Unit vector

            distsword2vec = dot(model3, meanword2vec)
            # The dot prodiuct of all possible labels in word2vec vocab with the unit vector of
            # topic word

            """
            This next section of code checks if the topic word is also a potential label in trained 
            word2vec model. If that is the case, it is important the dot product of label with that 
            topic word is not taken into account.Hence we make that zero and further down the code 
            also exclude it in taking average of that label over all topic words. 
            """
            if model2.vocab[item_].index in w_indices:
                i_val = w_indices.index(model2.vocab[item_].index)
                store_indices.append(i_val)
                distsword2vec[i_val] = 0.0
            valword2vec = valword2vec + distsword2vec

    avgdoc2vec = valdoc2vec / float(len(topic_list[topic_num]))
    # Give the average vector over all topic words
    avgword2vec = valword2vec / float(len(topic_list[topic_num]))
    # Average of word2vec vector over all topic words

    bestdoc2vec = matutils.argsort(avgdoc2vec, topn=100, reverse=True)
    # argsort and get top 100 doc2vec label indices
    resultdoc2vec = []
    # Get the doc2vec labels from indices
    for elem_ in bestdoc2vec:
        ind = d_indices[elem_]
        temp = model1.docvecs.index_to_doctag(ind)
        resultdoc2vec.append((temp, float(avgdoc2vec[elem_])))

    # This modifies the average word2vec vector for cases in which the word2vec label was same as
    # topic word.
    for element in store_indices:
        avgword2vec[element] = (avgword2vec[element] * len(topic_list[topic_num])) / (
            float(len(topic_list[topic_num]) - 1))

    bestword2vec = matutils.argsort(avgword2vec, topn=100,
                                    reverse=True)  # argsort and get top 100 word2vec label indices
    # Get the word2vec labels from indices
    resultword2vec = []
    for element in bestword2vec:
        ind = w_indices[element]
        temp = model2.index2word[ind]
        resultword2vec.append((temp, float(avgword2vec[element])))

    # Get the combined set of both doc2vec labels and word2vec labels
    comb_labels = list(set([j[0] for j in resultdoc2vec] + [k[0] for k in resultword2vec]))
    newlist_doc2vec = []
    newlist_word2vec = []

    # Get indices from combined labels 
    for elem_ in comb_labels:
        try:
            newlist_doc2vec.append(d_indices.index(model1.docvecs.doctags[elem_].offset))
            temp = get_word(elem_)
            newlist_word2vec.append(w_indices.index(model2.vocab[temp].index))
        except:
            pass
    newlist_doc2vec = list(set(newlist_doc2vec))
    newlist_word2vec = list(set(newlist_word2vec))

    # Finally again get the labels from indices. We searched for the score from both doctvec and word2vec
    # models
    resultlist_doc2vecnew = [
        (model1.docvecs.index_to_doctag(d_indices[elem_]), float(avgdoc2vec[elem_]))
        for elem_ in newlist_doc2vec
    ]
    resultlist_word2vecnew = [
        (model2.index2word[w_indices[elem_]], float(avgword2vec[elem_]))
        for elem_ in newlist_word2vec
    ]

    # Finally get the combined score with the label. The label used will be of doc2vec not of word2vec. 
    new_score = []
    for item_ in resultlist_word2vecnew:
        k, v = item_
        for elem_ in resultlist_doc2vecnew:
            k2, v2 = elem_
            k3 = get_word(k2)
            if k == k3:
                v3 = v + v2
                new_score.append((k2, v3))
    new_score = sorted(new_score, key=lambda x: x[1], reverse=True)
    return new_score[:(int(args.num_cand_labels))]


# cores = 4
# pool = mp.Pool(processes=cores)
# result = pool.map(get_labels, range(0, len(topic_list)))
# *** Error in `/home/andreas/bin/anaconda3/envs/gensim2/bin/python':
# corrupted double-linked list (not small): 0x0000559234a9d0a0 ***
# > trying single-threaded
result = map(get_labels, topic_list)

# The output file for candidates.
g = open(args.outputfile_candidates, 'w')
for i, elem in enumerate(result):
    val = ""
    for item in elem:
        val = val + " " + item[0]
    g.write(val + "\n")
g.close()

print("Candidate labels written to " + args.outputfile_candidates)
print("\n")
