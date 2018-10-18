# -*- coding: utf-8 -*-

"""
Author:         Shraey Bhatia
Date:           October 2016
File:           doc2vectrain.py

Gives a trained document vectors model (also known as Doc2VecModel). 
The input format are documents extrated using wiki extractor and tokenized using stanford tokenizer
stored in a directory
Output would be a trained doc2vec model.
Parameters taken for main_train.py
"""

import os
import re
import argparse
import unicodedata
import multiprocessing
from itertools import islice

from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# These are the arguments that are passed in main_train.py
parser = argparse.ArgumentParser()
parser.add_argument("epochs")  # Number of training epochs
parser.add_argument("input_dir")
parser.add_argument("output_dir")
args = parser.parse_args()


# Doc2vec Input documents Class. It uses yield to optimize memory usage and "tag" is Document title
# with an undescore.
class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

    def __iter__(self):

        for source in self.sources:
            with open(source, "r", encoding="utf-8") as fin:
                values = None
                found = None
                for cnt, line in enumerate(fin):
                    if "<doc" in line:  # Every new document starts with this format

                        # This gives the document title of Wikipedia
                        m = re.search('title="(.*)">', line)
                        if m:
                            found = m.group(1)
                            # found = found.lower()
                            found = unicodedata.normalize("NFKD", found)
                            found = found.replace(" ", "_")
                            found = found.encode('utf-8')
                        else:
                            found = None
                        values = []
                    else:
                        # </doc tells us end of document, till not reached it is same document
                        if "</doc" not in line:
                            for word in line.split(" "):
                                values.append(word.strip())
                        if "</doc" in line:
                            if found is not None:
                                yield LabeledSentence(words=values, tags=[found])


#cores = multiprocessing.cpu_count()
cores = 3
filenames = []


for path, subdirs, files in os.walk(args.input_dir):
    for name in files:
        temp = os.path.join(path, name)
        filenames.append(temp)

sentences = LabeledLineSentence(filenames)

for s in islice(sentences, 100):
    print(s)

quit()
# Doc2Vec model initialization and parameters
model = Doc2Vec(vector_size=300, window=15, min_count=20, sample=1e-5, workers=cores, hs=0, dm=0, negative=5,
                dbow_words=1, dm_concat=0)
model.build_vocab(sentences)

# Model Training
# for epoch in range(int(args.epochs)):
model.train(sentences, total_examples=2215485, epochs=(int(args.epochs)))
# print "Epoch completed: "+str(epoch+1)

model.save(output)
# supplied example count (2215485) did not equal expected count (2215484)
