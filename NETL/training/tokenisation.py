""" 
Author:         Shraey Bhatia
Date:           October 2016
File:           tokenisation.py

It takes in processed xml dump extraced by WikiExtractor and tokenises it using standford-parser for tokenization. 
You can use any of the below URL to download it and unzip it if want to run on your own.

http://nlp.stanford.edu/software/stanford-parser-full-2014-08-27.zip

The arguments for this file are given in main_train.py. 
"""

import os
import argparse
import sys

# Get the arguments passed in main_train.py
parser = argparse.ArgumentParser()
parser.add_argument("parser_loc")  # location of stanford parser giben in main_train.py
parser.add_argument("input_dir")  # Input diretory which is output of wiki-extractor processed xml dump
parser.add_argument("output_dir")  # Output directory for tokenised file
args = parser.parse_args()

# Checks if the output directory specified already exists. If it does removes it.

if os.path.isdir(args.output_dir):
    del_query = "rm -r " + args.output_dir
    os.system(del_query)

# Gets all the sub directories from the location
list_files = os.listdir(args.input_dir)

# Gets the classpath to run stanford tokenizer
classpath = args.parser_loc + "/stanford-parser.jar"

query1 = "mkdir " + args.output_dir
os.system(query1)
for item in list_files:
    if os.path.isdir(args.input_dir + "/" + item):
        inp_subdir = args.input_dir + "/" + item  # Getting the full path for subdirectories which needs to be tokenized.
        subfiles = os.listdir(inp_subdir)  # listing the files in subdirectory
        out_subdir = args.output_dir + "/" + item
        query = "mkdir " + out_subdir  # making new sub directories in output location, so that the directory structure of tokenised file is same as input directory
        os.system(query)

        for elem in subfiles:
            input_file = inp_subdir + "/" + elem  # Working on files in subdirectory. We need to tokenize them
            output_file = out_subdir + "/" + elem
            # query2 = "java -cp "+ classpath +" edu.stanford.nlp.process.PTBTokenizer -preserveLines --lowerCase <"+input_file+"> "+output_file # Java commanf to stanford tokenizer
            query2 = "java -cp " + classpath + " edu.stanford.nlp.process.PTBTokenizer -preserveLines <" + input_file + "> " + output_file  # Java commanf to stanford tokenizer
            print "Executing query"
            print query2
            os.system(query2)
