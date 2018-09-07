# -*- coding: utf-8 -*-

from os.path import join
import pandas as pd
from tabulate import tabulate

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
TOKEN = 'token'

### --- additional constants

# tags
ADJ = 'ADJ'
ADV = 'ADV'
INTJ = 'INTJ'
NOUN = 'NOUN'
PROPN = 'PROPN'
VERB = 'VERB'

ADP = 'ADP'
AUX = 'AUX'
CCONJ = 'CCONJ'
CONJ = 'CONJ'
DET = 'DET'
NUM = 'NUM'
PART = 'PART'
PRON = 'PRON'
SCONJ = 'SCONJ'

PUNCT = 'PUNCT'
SYM = 'SYM'
X = 'X'

SPACE = 'SPACE'
PHRASE = 'PHRASE'

# keys
IWNLP = 'IWNLP'
POS = 'POS'
INDEX = 'index'
START = 'start'
LEMMA = 'lemma'
TAG = 'tag'
STOP = 'stop'
ENT_TYPE = 'ent_type'
ENT_IOB = 'ent_iob'
KNOWN = 'known'
SENT_IDX = 'sent_idx'
SENT_START = 'sent_start'

### --- possible arguments ---

HPC = False
LOG = False
file_prefix = ""
DE = 'de'


def tprint(df: pd.DataFrame, head=0, to_latex=False):
    if head > 0:
        df = df.head(head)
    elif head < 0:
        df = df.tail(-head)
    print(tabulate(df, headers="keys", tablefmt="pipe") + '\n')

    if to_latex:
        print(df.to_latex(bold_rows=True))
