# -*- coding: utf-8 -*-
import re
from itertools import chain
from os.path import join


# --- default constants definitions ---

# - default paths -
DATA_BASE = "../data"
etl_base = "preprocessed"
ETL_PATH = join(DATA_BASE, etl_base)
LOCAL_PATH = etl_base
FULL_PATH = join(DATA_BASE, LOCAL_PATH)
nlp_base = "preprocessed/nlp"
NLP_PATH = join(DATA_BASE, nlp_base)
simple_base = "preprocessed/simple"
SIMPLE_PATH = join(DATA_BASE, simple_base)
tmp_base = "preprocessed/tmp"
TMP_PATH = join(DATA_BASE, tmp_base)
SPACY_PATH = join(NLP_PATH, 'spacy_model')
VOC_PATH = join(SPACY_PATH, 'vocab')
LDA_PATH = join(ETL_PATH, 'LDA_model')
LSI_PATH = join(ETL_PATH, 'LSI_model')
EMB_PATH = join(ETL_PATH, 'embeddings')
TPX_PATH = join(LDA_PATH, 'noun', 'bow', 'topics')

# - data scheme -
DATASET = 'dataset'
SUBSET = 'subset'
TIME = 'date_time'
ID = 'doc_id'
ID2 = 'doc_sub_id'
TITLE = 'title'
# AUTHOR -> mostly unknown or pseudonym
# SUBTITLE -> use DESCRIPTION
# CATEGORY -> use DESCRIPTION or LINKS
META = [DATASET, SUBSET, TIME, ID, ID2, TITLE]

TEXT = 'text'
# The DESCRIPTION and LINKS fields were introduced with dewiki and are so far unused in the other
# datasets.
# DESCRIPTION: Would be nice to add this especially to the news sites datasets.
# In dewiki it contains rather a subtitle than a description.
DESCRIPTION = 'description'
# LINKS: could be used to link forum threads together, although this is probably already done
# via ID[2]
LINKS = 'links'
# moved from META to DATA => hashes have to be recalculated!
TAGS = 'tags'
DATA = [DESCRIPTION, TEXT, LINKS, TAGS]

HASH = 'hash'
TOKEN = 'token'

# --- additional constants --

# Universal tagset
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

# additional
SPACE = 'SPACE'
PHRASE = 'PHRASE'
NER = 'NER'
NPHRASE = 'NPHRASE'

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


# --- for LDA modeling ---

# available datasets (unique)
DATASETS = {
    'dewac': 'dewac',
    'dewac1': 'dewac1',
    'dewiki': 'dewiki',
    'E': 'Europarl',
    'FA': 'FAZ',
    'FO': 'FOCUS',
    'N': 'news',
    'O': 'OnlineParticipation',
    'P': 'PoliticalSpeeches',
    'S': 'speeches',
}
# additional keys for available datasets
DATASETS_FULL = DATASETS.copy()
DATASETS_FULL.update({
    'dewi': 'dewiki',
    'dewik': 'dewiki',
    'dewa': 'dewac',
    'dewa1': 'dewac1',
    'e': 'Europarl',
    'europarl': 'Europarl',
    'fa': 'FAZ',
    'faz': 'FAZ',
    'fo': 'FOCUS',
    'focus': 'FOCUS',
    'o': 'OnlineParticipation',
    'onlineparticipation': 'OnlineParticipation',
    'p': 'PoliticalSpeeches',
    'politicalspeeches': 'PoliticalSpeeches',
    'n': 'news',
    'f': 'news',
    'F': 'news',
    's': 'speeches',
})

METRICS = ('ref', 'u_mass', 'c_v', 'c_uci', 'c_npmi', 'vote')
PARAMS = ('a42', 'b42', 'c42', 'd42', 'e42')
NB_TOPICS = (10, 25, 50, 100)
VERSIONS = ('noun', 'noun-verb', 'noun-verb-adj')
CORPUS_TYPE = ('bow', 'tfidf')

# --- filter lookup table ---

# the following tokens are filtered before applying LDA training
BAD_TOKENS_DICT = {
    'Europarl': [
        'E.', 'Kerr', 'The', 'la', 'ia', 'For', 'Ieke', 'the', 'WPA', 'INSPIRE', 'EN', 'ASEM',
        'ISA', 'EIT',
    ],
    'FAZ_combined': [
        'S.', 'j.reinecke@faz.de', 'B.', 'P.', 'of',
    ],
    'FOCUS_cleansed': [
        'OTS', 'RSS', 'of', 'UP', 'v.',
    ],
    'OnlineParticipation': [
        'Re', '@#1', '@#2', '@#3', '@#4', '@#5', '@#6', '@#7', '@#8', '@#9', '@#1.1', 'Für',
        'Muss',
        'etc', 'sorry', 'Ggf', 'u.a.', 'z.B.', 'B.', 'stimmt', ';-)', 'lieber', 'o.', 'Ja',
        'Desweiteren', '@#4.1.1'
    ],
    'PoliticalSpeeches': [
        'ZIF', 'of', 'and', 'DFFF',
    ],
    'dewiki': [],
    'dewac': [
        'H.', 'm.', 'W.', 'K.', 'g.', 'r.', 'A.', 'f.', 'l.', 'J.', 'EZLN', 'LAGH', 'LSVD', 'AdsD',
        'NAD', 'DÖW', 'Rn',
    ],
}
BAD_TOKENS = set(chain(*BAD_TOKENS_DICT.values()))
PLACEHOLDER = '[[PLACEHOLDER]]'
MINIMAL_PATTERN = re.compile(r'.\.')
NOUN_PATTERN = re.compile(r'^([0-9]+.*?)*?[A-Za-zÄÖÜäöüß].*')
POS_N = [NOUN, PROPN, NER, NPHRASE]
POS_NV = [NOUN, PROPN, NER, NPHRASE, VERB]
POS_NVA = [NOUN, PROPN, NER, NPHRASE, VERB, ADJ, ADV]

"""
 list of tokens to ignore when at the beginning of a phrase
 This is needed to avoid changing all appearances of for example
 'die Firma' to 'Die_Firma' since this is also a movie title.
"""
BAD_FIRST_PHRASE_TOKEN = {
    'ab', 'seit', 'in', 'der', 'die', 'das', 'an', 'am', 'diese', 'bis', 'ein', 'es', 'mit', 'im',
    'für', 'zur', 'auf', '!', '(', 'ich', 'so', 'auch', 'wir', 'auch', 'mich', 'du',
}

GOOD_IDS = {
    'dewac': join(ETL_PATH, 'dewac_good_ids.pickle'),
    'dewiki': join(ETL_PATH, 'dewiki_good_ids.pickle'),
}
