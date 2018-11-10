# coding: utf-8

"""
This processing step is also included in phrase_extraction_in_batches.py leading to some redundancy.
For some documents it had to be applied seperately and was first developed here and later added to
the simple/phrase-extraction pipeline. This should be streamlined to avoid additional maintenance.
"""

from constants import *
from os import listdir, makedirs
from os.path import isfile, join, exists
import gc
import pandas as pd
import re
from tqdm import tqdm
from phrase_extraction_in_batches import insert_wikipedia_phrases
import argparse
tqdm.pandas()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--start", type=int, required=False, default=0)
parser.add_argument("--nbfiles", type=int, required=False, default=None)
args = vars(parser.parse_args())
start = args['start']
nbfiles = args['nbfiles']
if nbfiles is not None:
    print(f'processing {nbfiles} files')
    nbfiles += start
dataset = args['dataset']
subdir = ''
if dataset.startswith('dewi'):
    subdir = 'dewiki'

goodids = None
pattern = re.compile(dataset)
files = sorted([f for f in listdir(join(SMPL_PATH, subdir))
                if (isfile(join(SMPL_PATH, subdir, f)) and pattern.match(f))])

for name in files[start:nbfiles]:
    gc.collect()
    corpus = name.split('.')[0]
    f = join(SMPL_PATH, subdir, name)
    print(corpus, f)
    df = pd.read_pickle(f)

    # filter Wikipedia documents. Process only valid articles.
    if name.startswith('dewiki'):
        if goodids is None:
            goodids = pd.read_pickle(join(ETL_PATH, 'dewiki_good_ids.pickle'))
        df = df[df.hash.isin(goodids.index)]

    df = insert_wikipedia_phrases(df)

    if name.startswith('dewiki'):
        f = join(SMPL_PATH, name)
    else:
        out_dir = join(SMPL_PATH, 'wiki_phrases')
        if not exists(out_dir):
            makedirs(out_dir)
        f = join(out_dir, f'{corpus}_wiki_phrases.pickle')
    print('Writing', f)
    df.to_pickle(f)
    gc.collect()
