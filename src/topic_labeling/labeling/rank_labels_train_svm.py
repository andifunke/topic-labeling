"""
Author:         Shraey Bhatia
Date:           October 2016
File:           train_svm_model.py

This file is to generate the trained svm model. You can specify your own datset. By default 
will take our dataset. Note a trained svm model for our datset is already in place in
model_run/support_files.
You will need binary svm_learn from SVM rank. URL probvided in readme. Update the path here if different.

(adapted and refactored to Python 3 and our current data scheme.
January 2019, Andreas Funke)
"""

import os
import re
from collections import defaultdict, Counter, OrderedDict
from os.path import join

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from topic_labeling.constants import DATA_BASE, DATASETS_FULL

pd.options.display.max_columns = 80
pd.options.display.max_rows = 100
pd.options.display.width = 1000


# Global parameters for the model.
ratings_version = ['all', 'removed_constants', 'cleaned_part', 'cleaned_full'][-1]
svm_path = join(DATA_BASE, 'ranker')
labels_path = join(svm_path, f'ratings_{ratings_version}.csv')  # label dataset with ratings

svm_learn_path = join(svm_path, 'svm_rank_learn')  # path to SVM rank trainer binary
pagerank_path = join(svm_path, 'pagerank-titles-sorted_de_categories_removed.txt')  # pagerank file
topics_path = join(svm_path, 'topics.csv')  # topic dataset with topic terms
svm_hyperparameter = 0.1  # The SVM hyperparameter.
omit_underscores = False  # alternative way to create trigrams (treating phrases as multiple tokens)
dsets = ['O', 'P', 'N', 'dewac']
datasets = [DATASETS_FULL.get(d, d) for d in dsets]
dstr = ('_'+'-'.join(dsets)) if dsets else ''
output_svm_model = join(svm_path, f'svm_model_{ratings_version}{dstr}')  # path for trained SVM model
tmp_file_path = join(svm_path, f"train_temp_{ratings_version}{dstr}.dat")

FEATURES = ['letter_trigram', 'prank', 'lab_length', 'common_words']


def normalize(item):
    """ Normalizes strings in topics and labels. """
    if isinstance(item, str):
        item = item.lower()
        for k, v in {' ': '_', '_!': '!', '_’': '’'}.items():
            item = item.replace(k, v)
        return item
    else:
        return item


def load_pageranks(file):
    """ Reading in pageranks and converting it into a dictionary. """
    f2 = open(file, 'r')
    p_rank_dict = {}
    for line in f2:
        word = line.split()
        try:
            assert len(word) == 2
        except:
            print(word)
        p_rank_dict[word[1].lower()] = word[0]
    print("PageRank model loaded")
    return p_rank_dict


def load_topics(file, datassetz=None):
    """ reading topic terms. """
    topics = pd.read_csv(file)
    if datassetz:
        topics = topics[topics.domain.isin(datassetz)]
        print(topics.head())
    topics = topics.applymap(normalize)
    topics = topics.drop('domain', axis=1, errors='ignore')
    topic_dict = topics.set_index('topic_id').T.to_dict('list')
    topic_ids = list(topics.index)
    return topic_dict, topic_ids


def load_labels(file, topic_ids, datassetz=None):
    """Reading topic labels."""
    labels = pd.read_csv(file)
    if datassetz:
        labels = labels[labels.topic_id.isin(topic_ids)]
        print(labels.head())
    labels.label = labels.label.apply(normalize)
    topic_labels_without_topic_id = list(labels)
    topic_labels_without_topic_id.remove('topic_id')
    labels['total'] = labels[topic_labels_without_topic_id].sum(axis=1)
    num_raters = labels.count(axis=1) - 3
    labels['avg'] = labels['total'] / num_raters
    topic_groups = labels.groupby('topic_id')

    labels_dict = OrderedDict()
    for tpxid, group in topic_groups:
        temp2 = []
        temp = list(group.label)
        for elem in temp:
            elem = elem.replace(" ", "_")
            temp2.append(elem)
        labels_dict[tpxid] = temp2
    return labels, labels_dict


def get_topic_lt(tpx):
    # Method to get letter trigrams for topic terms.
    tot_list = []
    for term in tpx:
        if omit_underscores:
            tokens = re.split(r'[_ ]', term)
            for token in tokens:
                trigrams = [token[i:i + 3] for i in range(0, len(token) - 2)]
                tot_list = tot_list + trigrams
        else:
            trigrams = [term[i:i + 3] for i in range(0, len(term) - 2)]
            tot_list = tot_list + trigrams
    counter = Counter(tot_list)
    total = sum(counter.values(), 0.0)
    for key in counter:
        counter[key] /= total
    return counter


def get_lt_ranks(lab_list, topic_list, num):
    """
    This method will be used to get first feature of letter trigrams for candidate labels and then
    rank them. It use cosine similarity to get a score between a letter trigram vector of label
    candidate and vector of topic terms.The ranks are given based on that score.
    """
    topic_ls = get_topic_lt(topic_list[num])
    val_list = []
    final_list = []
    for term in lab_list:
        if omit_underscores:
            # ignores underscores and spaces
            tokens = re.split(r'[_ ]', term.lower())
            trigrams = []
            for token in tokens:
                trigrams += [token[i:i + 3] for i in range(0, len(token) - 2)]
        else:
            trigrams = [term[i:i + 3] for i in
                        range(0, len(term) - 2)]  # Letter trigram for candidate label.
        label_cnt = Counter(trigrams)
        total = sum(label_cnt.values(), 0.0)
        for key in label_cnt:
            label_cnt[key] /= total
        tot_keys = set((list(topic_ls.keys()) + list(label_cnt.keys())))
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
        val_list.append((term, val))
    rank_val = [i[1] for i in val_list]
    arr = np.array(rank_val)
    order = arr.argsort()
    ranks = order.argsort()
    for i, elem in enumerate(val_list):
        final_list.append((elem[0], ranks[i], int(num)))

    return final_list


def generate_lt_feature(labels_list, topic_dict):
    """ Generates letter trigram feature """
    temp_lt = []
    for k, v in topic_dict.items():
        temp_lt.append(get_lt_ranks(labels_list[k], topic_dict, k))
    lt_feature = [item for sublist in temp_lt for item in sublist]
    print("Letter trigram feature generated")
    return lt_feature


def change_format(f1):
    """ Changes the format of letter trigram into a dict of dict. """
    lt_dict = defaultdict(dict)
    for elem in f1:
        x, y, z = elem
        lt_dict[z][x] = y
    return lt_dict


def prepare_features(letter_tg_dict, page_rank_dict, topic_list, labels=None):
    """
    This method is to prepare all features. It will take in dictionary of letter trigram, pagerank,
    list of all columns for the datframe and name of features. It will generate four features in the
    dataframe namely Pagerank, letter trigram, Topic overlap and Number of words in a label.
    Additionally DataFrame will also be given the label name, topic_id and an avg_val which is average
    annotator value. This annotator avlue is calculated from the candidate label datset and is used to
    train the SVM model.
    """
    cols = ['label', 'topic_id', 'letter_trigram', 'prank', 'lab_length', 'common_words', 'avg_val']
    frame = pd.DataFrame()

    for idx, a in letter_tg_dict.items():
        temp_frame = pd.DataFrame()
        for t_label in a:
            new_list = []  # The list created to get values for dataframe.
            new_list.append(t_label)  # Candidate label name
            new_list.append(idx)  # Topic_id
            temp_val = a[t_label]  # letter trigram feature
            new_list.append(temp_val)

            # --- Page Rank Feature ---
            try:
                pagerank = page_rank_dict[t_label]
                pagerank = float(pagerank.replace(',', '.'))
            except Exception as e:
                print('not in pagerank file:', e)
                pagerank = np.nan
            new_list.append(pagerank)

            # --- Topic overlap feature ---
            word_labels = t_label.split("_")
            overlap = set(word_labels).intersection(set(topic_list[idx]))
            com_word_length = len(overlap)

            # Num of words in candidate label feature
            lab_length = len(word_labels)
            new_list.append(lab_length)
            new_list.append(com_word_length)

            # of labels are provided the features will be prepared for training
            if labels is not None:
                # The annotator value.
                mask = (labels.topic_id == idx) & (labels.label == t_label)
                val = labels.loc[mask, 'avg'].values[0]
                new_list.append(val)
            else:
                # This could be just any value appended for the sake of giving a column for annotator
                # rating neeeded in SVM Ranker classify
                new_list.append(3)

            temp = pd.Series(new_list, index=cols)
            temp_frame = temp_frame.append(temp, ignore_index=True)
            temp_frame = temp_frame.fillna(0)

        for item in FEATURES:
            # Feature normalization per topic.
            temp_frame[item] = (temp_frame[item] - temp_frame[item].mean()) / \
                               (temp_frame[item].max() - temp_frame[item].min())
        frame = frame.append(temp_frame, ignore_index=True)
    frame = frame.fillna(0)
    return frame


def convert_dataset(dataset):
    """ converts the dataset into a format which is taken by SVM ranker. """
    data_lst = []
    for i in range(len(dataset)):

        mystring = str(dataset[i:i + 1]["avg_val"].values[0]) + " " + "qid:" + str(
            int(dataset[i:i + 1]["topic_id"].values[0]))
        for j, item in enumerate(FEATURES):
            mystring = mystring + " " + str(j + 1) + ":" + str(dataset[i:i + 1][item].values[0])
        mystring = mystring + " # " + dataset[i:i + 1]['label'].values[0]
        data_lst.append(mystring)
    return data_lst


def train(train_set, tmp_file, svm_learn_file):
    """ This method generates the trained SVM file using SVM ranker learn """
    with open(tmp_file, "w") as fp:
        for item in train_set:
            fp.write("%s\n" % item)

    query = ' '.join([
        svm_learn_file,
        '-c',
        str(svm_hyperparameter),
        tmp_file,
        output_svm_model
    ])
    print(query)
    print()
    os.system(query)

    query2 = f'rm {tmp_file}'
    print()
    print(query2)
    os.system(query2)


def main():
    topics_dict, topic_ids = load_topics(topics_path, datasets)
    labels, labels_dict = load_labels(labels_path, topic_ids, datasets)
    p_rank_dict = load_pageranks(pagerank_path)
    letter_trigram_feature = generate_lt_feature(labels_dict, topics_dict)
    lt_dict = change_format(letter_trigram_feature)
    feature_dataset = prepare_features(lt_dict, p_rank_dict, topics_dict, labels=labels)
    print("All features generated")
    train_list = convert_dataset(feature_dataset)
    train(train_list, tmp_file_path, svm_learn_path)


if __name__ == '__main__':
    main()
