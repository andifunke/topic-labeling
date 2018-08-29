import os
import pickle
from pprint import pprint

### --- constants definitions ---

DATA_BASE = "../../master_cloud/corpora"
ETL_BASE = "preprocessed"
ETL_PATH = os.path.join(DATA_BASE, ETL_BASE)

# standard meta data fields
DATASET = 'dataset'
SUBSET = 'subset'
ID = 'doc_id'
ID2 = 'doc_subid'
TITLE = 'doc_title'
CAT = 'doc_category'

CORPUS = "OnlineParticipation"


def load(name):
    file_path = os.path.join(ETL_PATH, CORPUS, name+'_data.pickle')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'rb') as fp:
        return pickle.load(fp)


meta = load('meta')
print(type(meta))
print(len(meta))
pprint(meta)


txt = load('text')
print(type(txt))
print(len(txt))
# pprint(txt)

