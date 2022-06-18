"""
Author:         Shraey Bhatia
Date:           October 2016
File:           unsupervised_labels.py

This file will take candidate labels and give the best labels from them using unsupervised way
which is just going to be based on letter trigram ranking.

(gently adapted to Python 3 and our current data scheme. We use one csv-file for topics and labels.
January 2019, Andreas Funke)
"""

import argparse
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from topiclabeling.utils import tprint

parser = argparse.ArgumentParser()
parser.add_argument("num_unsup_labels")  # The number of unsupervised labels.
parser.add_argument("tdata")  # The topic data file. It contains topic terms.
parser.add_argument("ldata")  # The topic data file. It contains topic terms.
parser.add_argument("output_unsupervised")  # The file in which output is written
args = parser.parse_args()

# Reading in the topic terms from the topics file.
topics = pd.read_csv(args.tdata, index_col=[0, 1])
index = topics.index.copy()

topics = topics.reset_index(drop=True)
topics = topics.applymap(str.lower)
labels = pd.read_csv(args.ldata, index_col=[0, 1]).reset_index(drop=True)
labels = labels.applymap(str.lower)
topic_cols = [col for col in topics.columns if "term" in col]
label_cols = [col for col in labels.columns if "label" in col]
topics = topics[topic_cols]
labels = labels[label_cols]
tprint(topics, 5)
tprint(labels, 5)

topic_list = topics.T.to_dict("list")
label_list = labels.values.tolist()

print("Data Gathered for unsupervised model")


# Method to get letter trigrams for topic terms.
def get_topic_lg(elem):
    tot_list = []
    for item in elem:
        trigrams = [item[i : i + 3] for i in range(0, len(item) - 2)]
        tot_list = tot_list + trigrams
    x = Counter(tot_list)
    total = sum(x.values(), 0.0)
    for key in x:
        x[key] /= total
    return x


def get_best_label(label_list, num):
    """
    This method will be used to get letter trigrams for candidate labels and then rank them. It uses
    cosine similarity to get a score between a letter trigram vector of label candidate and vector of
    topic terms.The ranks are given based on that score. Based on this rank It will give the best
    unsupervised labels.
    """
    topic_ls = get_topic_lg(topic_list[num])
    val_dict = {}
    values = []
    for item in label_list:
        trigrams = [
            item[i : i + 3] for i in range(0, len(item) - 2)
        ]  # Extracting letter trigram for label
        label_cnt = Counter(trigrams)
        total = sum(label_cnt.values(), 0.0)
        for key in label_cnt:
            label_cnt[key] /= total
        tot_keys = list(set(list(topic_ls.keys()) + list(label_cnt.keys())))
        listtopic = []
        listlabel = []
        for elem in tot_keys:
            if elem in topic_ls:
                listtopic.append(topic_ls[elem])
            else:
                listtopic.append(0.0)
            if elem in label_cnt:
                listlabel.append(label_cnt[elem])
            else:
                listlabel.append(0.0)
        val = 1 - cosine(np.array(listtopic), np.array(listlabel))  # Cosine Similarity
        val_dict[item] = val
        values.append(val)
    # print(label_list)
    # print(val_dict)
    # print(values)
    # list_sorted = sorted(val_dict.items(), key=lambda x: x[1],
    #                      reverse=True)  # Sorting the labels by rank
    return values  # [i[0] for i in list_sorted[:int(args.num_unsup_labels)]]


unsup_output = []
for j in range(0, len(topic_list)):
    unsup_output.append(get_best_label(label_list[j], j))

dfout = pd.DataFrame(unsup_output)
dfout.index = index
print(dfout)
dfout.to_csv("../data/unsup_output.csv")

# printing the top unsupervised labels.
# print("Printing labels for unsupervised model")
# print("\n")
# g = open(args.output_unsupervised, 'w')
# for i, item in enumerate(unsup_output):
#     all_values.append()
# print("Top " + args.num_unsup_labels + " labels for topic " + str(i) + " are:")
# g.write("Top " + args.num_unsup_labels + " labels for topic " + str(i) + " are:" + "\n")
# for elem in item:
# print(elem)
# g.write(elem + "\n")
# print("\n")
# g.write("\n")
# g.close()
