# -*- coding: utf-8 -*-

from os.path import join

### --- default constants definitions ---

## default paths
DATA_BASE = "../../master_cloud/corpora"
ETL_BASE  = "preprocessed"
ETL_PATH  = join(DATA_BASE, ETL_BASE)
# TODO: add local path to arguments/options
LOCL_PATH = ETL_BASE
FULL_PATH = join(DATA_BASE, LOCL_PATH)
NLP_BASE  = "preprocessed/nlp"
NLP_PATH  = join(DATA_BASE, NLP_BASE)
SPCY_PATH = join(NLP_PATH, 'spacy_model')
VOC_PATH  = join(SPCY_PATH, 'vocab')

## data scheme
DATASET   = 'dataset'
SUBSET    = 'subset'
TIME      = 'date_time'
ID        = 'doc_id'
ID2       = 'doc_subid'
TITLE     = 'title'
# AUTHOR -> mostly unknown or pseudonym
# SUBTITLE -> use DESCRIPTION
# CATEGORY -> use DESCRIPTION or LINKS
META = [DATASET, SUBSET, TIME, ID, ID2, TITLE]

TEXT      = 'text'
# The DESCRIPTION and LINKS fields were introduced with dewiki and are so far unused in the other datasets.
# DESCRIPTION: Would be nice to add this especially to the news sites datasets.
# In dewiki it contains rather a subtitle than a description.
DESCR     = 'description'
# LINKS: could be used to link forum threads together, although this is probably already done via ID[2]
LINKS     = 'links'
# moved from META to DATA => hashes have to be recalculated!
TAGS      = 'tags'
DATA = [DESCR, TEXT, LINKS, TAGS]

HASH      = 'hash'
TOKEN     = 'token'


### --- additional constants

# Universal Tagset
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
TOK_IDX = 'tok_idx'
START = 'start'
LEMMA = 'lemma'
TAG = 'tag'
STOP = 'stop'
ENT_TYPE = 'ent_type'
ENT_IOB = 'ent_iob'
ENT_IDX = 'ent_idx'
KNOWN = 'known'
SENT_IDX = 'sent_idx'
SENT_START = 'sent_start'
NOUN_PHRASE = 'noun_phrase'
