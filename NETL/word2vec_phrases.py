# -*- coding: utf-8 -*-

"""
Author:         Shraey Bhatia
Date:           October 2016
File:           word2vec_phrases.py

This file give us a list of Wikipedia titles which has to be used as ngrams in running word2vec model.
This removes brackets from title and all filter with ttile length and document size. This is the file
to be used in ngramsgen.py. Stanford Parser is used to tokenise the files
as replacement has to be occured on tokenised files with the same tokeniser. 
The output of this file is also used in generating output of word2vec_indices.py which is used in
model_run/candidate-generation.py.
One such file; URL is in readme Word2vec Phrase list. You can update parameters in this file.
"""

import re
import os
import multiprocessing as mp
from multiprocessing import Pool

# Gobals
# Length of wikipedia title you want to filter. All titles greter than or equal to this value will be
# thrown away
title_length = 5
# length of documents. All documents having number of words less than this value will not be considered.
doc_length = 40
# the directory in which you tokenised all files extracted from Wiki-Extractor using stanford tokenizer
tokenised_wiki_directory = 'training/processed_documents/docs_tokenised'
# The name of output file you want the list of valid wikipedia titles to be saved into. Will be a pickle
# file
output_filename = 'training/additional_files/word2vec_phrases_list_tokenized2.txt'
# Directory of stanford Parser.
loc_parser = "training/support_packages/stanford-parser-full-2018-02-27"
# Full classpath for jar file
classpath = loc_parser + "/stanford-parser.jar"


# Method removes parenthesis brackets from labels
def get_word(word):
    inst = re.search(r" \((.+)\)", word)
    if inst is None:
        return word
    else:
        word = re.sub(r' \(.+\)', '', word)
        return word


def get_labels(filename):
    list_labels = []
    f = open(filename, "r")
    for line in f:
        if "<doc" in line:
            # Uses regular expression to get wiki title. The title is in this format if you use Wiki-Extractor.
            m = re.search('title="(.*)">', line)
            try:
                found = m.group(1)
            except:
                found = ""
            values = []
        else:
            if found != "":
                if "</doc" not in line:
                    for word in line.split(" "):
                        values.append(word.strip())

                # checks if we reach end of that particular document and if condition ois satisfied
                # title is added into list.
                if "</doc" in line:
                    temp_list = found.split(" ")
                    if (len(values) > doc_length) and (len(temp_list) < title_length):
                        found = get_word(found)  # Removing brackets if present

                list_labels.append(found)
    return list_labels


# Walking through directory and getting the filenames from the tokenised directory.
filenames = []
for path, subdirs, files in os.walk(tokenised_wiki_directory):
    for name in files:
        temp = os.path.join(path, name)
        filenames.append(temp)
filenames = sorted(filenames)
# filenames_temp = filenames[:4]
print("Got all files")
print(filenames)

# Multiprocess files
cores = mp.cpu_count()
pool = Pool(processes=cores)
y_parallel = pool.map(get_labels, filenames)

# converting a list of list into list
all_docs = [item for sublist in y_parallel for item in sublist]

# Writing list of titles into temporary file which will be okenised
print("Generating a temporary file")

g = open("temp.txt", 'w')
set_docs = sorted(set(all_docs))
bad_encodings = {'₩', '₡'}
for elem in set_docs:
    if elem in bad_encodings:
        print(elem)
        continue
    g.write(elem + '\n')
g.close()

print("Tokenising")

# Running query for standford parser
query = ' '.join([
    "java -cp",
    classpath,
    "edu.stanford.nlp.process.PTBTokenizer -preserveLines -encoding UTF-8 <temp.txt>",
    output_filename,
])
print(query)
os.system(query)

# Deleting temporary file
os.system("rm temp.txt")
