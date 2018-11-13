"""
Author:         Shraey Bhatia
Date:           October 2016
File:           get_indices.py

This file takes in the output of pruned_documents.py and word2vec_phrases.py and give back the
respective indices from doc2vec model and word2vec model respectively. You can also download these
output files from URLs in readme.(Word2vec phrase List and Filtered/short document titles). Though
all these files have already been given to run the models but script is given if you want to create
your own. These indices fileswill be used in cand-generation.py to generate label candidates.
"""
from itertools import islice

from gensim.models import Doc2Vec
from gensim.models import Word2Vec
import re
import pickle

# Global Parameters
# Trained Doc2vec Model
doc2vec_model = "model_run/pre_trained_models/doc2vec_en/docvecmodel.d2v"
# Trained word2vec model
word2vec_model = "model_run/pre_trained_models/word2vec_en/word2vec"
# The file created by pruned_documents.py. FIltering short or long title documents.
short_label_documents = "additional_support_files/short_label_documents_en"
# The file created by word2vec_phrases.py Removing brackets from filtered wiki titles.
short_label_word2vec_tokenised = "training/additional_files/word2vec_phrases_list_tokenized2.txt"
# The output file which map pruned doc2vec labels to indcies from doc2vec model.
doc2vec_indices_output = "doc2vec_indices"
# the output file that maps short_label_word2vec_tokenised to indices from wrod2vec model.
word2vec_indices_output = "word2vec_indices"


# Removing any junk labels and also if a label pops up with the term disambiguation.
def get_word(word):
    inst = re.search(r"_\(([A-Za-z0-9_]+)\)", word)

    if inst is None:
        length = len(word.split("_"))
        if length < 5:
            return True, word
    else:
        if inst.group(1) != "disambiguation":
            word2 = re.sub(r'_\(.+\)', '', word)
            if len(word2.split(" ")) < 5:
                return True, word

    return False, word


# Load the trained doc2vec and word2vec models.
print('loading d2v')
d2v = Doc2Vec.load(doc2vec_model)
print('loading w2v')
w2v = Word2Vec.load(word2vec_model)
print("Models loaded")

# Loading the pruned tiles and making a set of it
with open(short_label_documents, "rb") as fp:
    doc_labels = pickle.load(fp)
doc_labels = set(doc_labels)
print(len(doc_labels))
print(sorted(doc_labels)[:20])

# laoding thw phrasses used in training word2vec model. And then replacing space with underscore.
with open(short_label_word2vec_tokenised, 'r') as fp:
    print("loaded", short_label_word2vec_tokenised)
    list_labels = []
    for line in fp:
        line = line.strip()
        list_labels.append(line)
    list_labels = set(list_labels)
print(len(list_labels))
print(sorted(list_labels)[:20])

word2vec_labels = []
for words in list_labels:
    new = words.split(" ")
    temp = '_'.join(new)
    word2vec_labels.append(temp)
word2vec_labels = set(word2vec_labels)
print("Word2vec model phrases loaded")

doc_indices = []
word_indices = []

print('doc_labels')
# finds the coresponding index of the title from doc2vec model
for elem in islice(doc_labels, 100):
    print(elem)
    status, item = get_word(elem)
    if status:
        try:
            val = d2v.docvecs.doctags[elem].offset
            print(val)
            doc_indices.append(val)
        except Exception as e:
            print('not found', e)

print('w2v_labels')
# Finds the corseponding index from word2vec model
for elem in islice(word2vec_labels, 100):
    print(elem)
    try:
        val = w2v.wv.vocab[elem].index
        print(val)
        word_indices.append(val)
    except Exception as e:
        print('not found', e)
quit()

# creating output indices file
with open(doc2vec_indices_output, 'wb') as fp:
    pickle.dump(doc_indices, fp)
with open(word2vec_indices_output, 'wb') as fp:
    pickle.dump(word_indices, fp)
