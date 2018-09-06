# coding: utf-8

# ### Default Imports

import spacy
import pandas as pd
import sys
import re
from os import listdir, makedirs
from os.path import isfile, join, exists
from iwnlp.iwnlp_wrapper import IWNLPWrapper


### --- default constants definitions ---

DATA_BASE = "../../master_cloud/corpora"
ETL_BASE = "preprocessed"
ETL_PATH = join(DATA_BASE, ETL_BASE)
NLP_BASE = "preprocessed/nlp"
NLP_PATH = join(DATA_BASE, NLP_BASE)
SPACY_PATH = join(NLP_PATH, 'spacy_model')
VOCAB_PATH = join(SPACY_PATH, 'vocab')

# standard meta data fields
DATASET = 'dataset'
SUBSET = 'subset'
ID = 'doc_id'
ID2 = 'doc_subid'
TITLE = 'title'
TAGS = 'tags'
TIME = 'date_time'
# AUTHOR
# SUBTITLE
# CATEGORY
META = [DATASET, SUBSET, ID, ID2, TITLE, TAGS, TIME]
TEXT = 'text'
HASH = 'hash'

### --- additional constants

# tags
PUNCT = 'PUNCT'
DET = 'DET'
PHRASE = 'PHRASE'

# keys
IWNLP = 'IWNLP'
POS = 'POS'
INDEX = 'index'
START = 'start'
NOUN = 'NOUN'
PROPN = 'PROPN'
LEMMA = 'lemma'
TAG = 'tag'
STOP = 'stop'
ENT_TYPE = 'ent_type'
ENT_IOB = 'ent_iob'
KNOWN = 'known'


### --- load spacy and iwnlp ---

if len(sys.argv) > 1 and sys.argv[1] == '--hpc':
    print('on hpc')
    de = '/home/funkea/.local/lib/python3.4/site-packages/de_core_news_sm/de_core_news_sm-2.0.0'
else:
    de = 'de'

print("loading spacy")
nlp = spacy.load(de)  # <-- load with dependency parser (slower)
# nlp = spacy.load(de, disable=['parser'])

if exists(VOCAB_PATH):
    print("reading vocab from", VOCAB_PATH)
    nlp.vocab.from_disk(VOCAB_PATH)

print("loading IWNLPWrapper")
lemmatizer = IWNLPWrapper(lemmatizer_path='../data/IWNLP.Lemmatizer_20170501.json')


### --- function definitions ---

def process_phrases(doc):
    """ 
        given a doc process and return the contained noun phrases.
        This function is based on spacy's noun chunk detection. 
        It also creates items for a global phrase lookup table, which are currently not used.
    """

    # clean the noun chuncs from spacy first
    noun_chunks = []
    for chunk in doc.noun_chunks:
        start = False
        noun_chunk = []
        for token in chunk:
            # exclude punctuation
            if token.pos_ == PUNCT:
                continue
            # exclude leading determiners
            if not start and (token.pos_ == DET or token.is_stop):
                continue
            start = True
            noun_chunk.append(token)
        if len(noun_chunk) > 1:
            noun_chunks.append(noun_chunk)
    
    # the remaining, adjusted noun chunks will be lemmatized and indexed
    phrase_list_lookup = []
    phrase_list_doc = []
    for chunk in noun_chunks:
        phrase = []
        for token in chunk:
            lemma, _ = lemmatize(token.text, token.pos_)
            if lemma:
                phrase.append(lemma)
            else:
                phrase.append(token.text)
        phrase = ' '.join(phrase)
        text = ' '.join([t.text for t in chunk])
        
        # add to phrase collection of corpus
        phrase_lookup = pd.Series()
        phrase_lookup['lemmatized'] = phrase
        phrase_lookup['original'] = text
        # phrase_lookup['Spacy Tokens'] = tuple(chunk)
        phrase_list_lookup.append(phrase_lookup)
        
        # add to document dataframe
        phrase_series = pd.Series()
        phrase_series[TEXT] = text
        phrase_series[IWNLP] = phrase
        phrase_series[POS] = PHRASE
        phrase_series[INDEX] = chunk[0].i
        phrase_series[START] = chunk[0].idx
        phrase_list_doc.append(phrase_series)

    # return the dataframes and for the doc dataframe and for the global phrase lookup table
    return pd.DataFrame(phrase_list_doc), pd.DataFrame(phrase_list_lookup)


def lemmatize(token: str, pos: str) -> (str, bool):
    """ 
    This function uses the IWNLP lemmatizer with a few enhancements for compund nouns and nouns 
    with uncommon capitalization. Can also be used to lemmatize tokens with different POS-tags.
    Do not use this function to lemmatize phrases.
    :param token: white space stripped single token (str)
    :param pos:   string constant, one of Universal tagset.
    :return: tuple of type (str, bool)
           value[0]: The lemma of the token if a lemma can be derived, else None.
           value[1]: True if the token can be retrieved from the Wiktionary database as is, else False.
    """
    
    if pos == PHRASE:
        try:
            raise ValueError
        except ValueError:
            print("Don't lemmatize Phrases with this function!")
    
    lemm = lemmatizer.lemmatize(token, pos)
    # default lemmatization ok?
    if lemm:
        return lemm[0], True

    # some rules to derive a lemma from the original token (nouns only)
    # TODO: define rules for hyphenated nouns
    if pos == NOUN or pos == PROPN:
        # first try default noun capitalization
        lemm = lemmatizer.lemmatize(token.title(), pos)
        if lemm:
            return lemm[0], False

    # still no results: try noun suffixes
        for i in range(1, len(token)-1):
            token_edit = token[i:].title()
            lemm = lemmatizer.lemmatize_plain(token_edit, ignore_case=True)
            if lemm:
                lemm = lemm[0]
                lemm = token[:i].title() + lemm.lower()
                return lemm, False
    
    # sorry, no results found:
    return None, False


def essence_from_doc(doc, key):
    """
    Creates a pandas DataFrame from a given spacy.doc that contains only nouns and noun phrases.
    :param doc: spacy.doc
    :return:     pandas.DataFrame
    """
    tags = [
        (
         token.text, token.lemma_, token.pos_, token.tag_, token.is_stop,
         token.i, token.idx,
         token.ent_type_, token.ent_iob_, # token.ent_id_,
         ) for token in doc]
    df = pd.DataFrame(tags)
    df = df.rename(columns={k:v for k,v in enumerate([
          TEXT, LEMMA, POS, TAG, STOP, INDEX, START, ENT_TYPE, ENT_IOB,
          # "Dep", "Shape", "alpha", "Ent_id"  # currently not used :(
    ])})
    
    # add IWNLP lemmatization
    df[IWNLP], df[KNOWN] = zip(*df.apply(lambda row: lemmatize(row[TEXT], row[POS]), axis=1))
    
    # add phrases
    df_phrases, phrase_lookup = process_phrases(doc)
    df = df.append(df_phrases).sort_values(START)
    df = df[df.POS.isin([NOUN, PROPN, PHRASE])].reset_index(drop=True)
    
    # replace Text with lemmatization, if lemmatization exists
    mask = ~df[IWNLP].isnull()
    df.loc[mask, TEXT] = df.loc[mask, IWNLP]
    
    # add hash-key
    df[HASH] = key
    
    return df[[HASH, INDEX, TEXT, POS]], phrase_lookup


def process_docs(series, size=None):
    """ main function for sending the dataframes from the ETL pipeline to the NLP pipeline """
    length = len(series)
    steps = 100
    step_len = 100//steps
    percent = length//steps
    done = 0
    for i, kv in enumerate(series[:size].iteritems()):
        if i % percent == 0:
            print("{:d}%: {:d} documents processed".format(done, i))
            done += step_len

        k, v = kv
        # build spacy doc
        doc = nlp(v)
        essential_token, phrase_lookup = essence_from_doc(doc, key=k)
        yield essential_token, phrase_lookup
        
        
def store(corpus, df):
    """returns the file path where the dataframe was stores"""
    makedirs(NLP_PATH, exist_ok=True)
    fname = join(NLP_PATH, corpus + '.pickle')
    print('saving', corpus, 'to', fname)
    df.to_pickle(fname)
    return fname


def read(f):
    """ reads a dataframe from pickle format """
    print("reading corpus from", f)
    return pd.read_pickle(f)


def read_process_load(file_path, corpus):
    df = read(file_path)
    print("processing", corpus)
    docs, phrase_lookups = zip(*[tple for tple in process_docs(df[TEXT], size=None)])
    docs = pd.concat(docs).reset_index(drop=True)
    phrase_lookups = pd.concat(phrase_lookups).reset_index(drop=True)

    store(corpus + '_nlp', docs)
    store(corpus + '_phrase_lookups', phrase_lookups)
    print("writing spacy model to disk:", NLP_PATH)
    # stored with each corpus, in case anythings goes wrong
    nlp.to_disk(SPACY_PATH)
    # nlp.vocab.to_disk(VOCAB_PATH)


### --- run ---

if __name__ == "__main__":

    LOCAL_PATH = ETL_BASE
    FULL_PATH = join(DATA_BASE, LOCAL_PATH)

    files = sorted([f for f in listdir(FULL_PATH) if isfile(join(FULL_PATH, f))])

    for name in files:
        corpus = re.split(r'\.|_', name)[0]
        fname = join(FULL_PATH, name)
        read_process_load(fname, corpus)

    print("done")
