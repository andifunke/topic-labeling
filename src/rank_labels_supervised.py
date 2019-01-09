"""
Author:         Shraey Bhatia
Date:           October 2016
File: 		supervised_labels.py

This python code gives the top supervised labels for that topic. The paramters needed are passed
through get_labels.py. It generates letter_trigram,pagerank, Topic overlap and num of words in
features. Then puts it into SVM classify format and finally uses the already trained supervised model
to make ranking predictions and get the best label. You will need SVM classify binary from SVM rank.
The URL is provided in readme.

(adapted and refactored to Python 3 and our current data scheme.
January 2019, Andreas Funke)
"""
from os.path import join

import numpy as np
import re
import os
import argparse
import pandas as pd

from constants import DATA_BASE, DSETS
from rank_labels_train_svm import load_topics, load_labels, load_pageranks, generate_lt_feature, \
    change_format, prepare_features, convert_dataset

parser = argparse.ArgumentParser()
parser.add_argument("num_top_labels")  # number of top labels
parser.add_argument("ratings_version")
args = parser.parse_args()

# Global parameters for the model.
ratings_version = args.ratings_version
svm_path = join(DATA_BASE, 'ranker')
topics_path = join(svm_path, 'topics.csv')
labels_path = join(svm_path, f'ratings_{ratings_version}.csv')
svm_model = join(svm_path, f'svm_model_{ratings_version}')
tmp_file_path = join(svm_path, f"test_temp_{ratings_version}.dat")
output_path = join(svm_path, f"supervised_labels_{ratings_version}")
svm_classify_path = join(svm_path, 'svm_rank_classify')
pagerank_path = join(svm_path, 'pagerank-titles-sorted_de_categories_removed.txt')
tesets = ['dewac']
datasets = [DSETS.get(d, d) for d in tesets]
testsets = ('_' + '-'.join(tesets)) if tesets else ''

trsets = ['N']
trainsets = ('_'+'-'.join(trsets)) if trsets else ''
svm_model = join(svm_path, f'svm_model_{ratings_version}{trainsets}')
output_path = join(
    svm_path, f"supervised_labels_{ratings_version}__testset{testsets}__trainset{trainsets}"
)


def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


def predict(test_set, pred_chunk_size, tmp_file, svm_classifier_file, svm_model_file, out_file):
    """ calls SVM classify and gets predictions for each topic. """
    with open(tmp_file, "w") as fp:
        for item in test_set:
            fp.write("%s\n" % item)

    query = ' '.join([
        svm_classifier_file,
        tmp_file,
        svm_model_file,
        "predictionstemp"
    ])
    print(query)
    print()
    os.system(query)

    h = open("predictionstemp")
    pred_list = []
    for line in h:
        pred_list.append(line.strip())
    h.close()

    pred_chunks = chunks(pred_list, pred_chunk_size)
    test_chunks = chunks(test_set, pred_chunk_size)
    df_pred = pd.DataFrame.from_records(pred_chunks)
    df_test = pd.DataFrame.from_records(test_chunks)
    df_pred.to_csv(out_file+'_pred.csv')
    df_test.to_csv(out_file+'_features.csv')
    list_max = []
    for j in range(len(pred_chunks)):
        max_sort = np.array(pred_chunks[j]).argsort()[::-1][:int(args.num_top_labels)]
        list_max.append(max_sort)

    print()
    print("Printing Labels for supervised model")
    g = open(out_file, 'w')
    for cnt, (x, y) in enumerate(zip(test_chunks, list_max)):
        # print("Top " + args.num_top_labels + " labels for topic " + str(cnt) + " are:")
        g.write("Top " + args.num_top_labels + " labels for topic " + str(cnt) + " are:" + "\n")
        for i2 in y:
            m = re.search('# (.*)', x[i2])
            # print(m.group(1))
            g.write(m.group(1) + "\n")
        # print()
        g.write("\n")
    g.close()

    query2 = f"rm {tmp_file} predictionstemp"
    print()
    print(query2)
    os.system(query2)


def main():
    print(datasets)
    topics_dict, topic_ids = load_topics(topics_path, datasets)
    _, labels_dict = load_labels(labels_path, topic_ids, datasets)
    p_rank_dict = load_pageranks(pagerank_path)
    letter_trigram_feature = generate_lt_feature(labels_dict, topics_dict)
    lt_dict = change_format(letter_trigram_feature)
    feature_dataset = prepare_features(lt_dict, p_rank_dict, topics_dict, labels=None)
    print("All features generated")
    test_list = convert_dataset(feature_dataset)
    nb_labels = len(list(labels_dict.items())[0][1])
    predict(test_list, nb_labels, tmp_file_path, svm_classify_path, svm_model, output_path)


if __name__ == '__main__':
    main()
