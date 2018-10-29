# coding: utf-8

from constants import *
from os import listdir
from os.path import isfile, join
import gc
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

from utils import tprint

tqdm.pandas()
pd.options.display.max_rows = 2001

p = pd.read_pickle(join(ETL_PATH, 'dewiki_phrases_joined.pickle'))
ps = set(p)

bad = {
    'ab', 'seit', 'in', 'der', 'die', 'das', 'an', 'am', 'diese', 'bis', 'ein'
}


def ngrams(ser):
    if ser[0].lower() not in bad:
        s = ser.str.cat(sep='_')
        size = len(ser)
        while size > 1:
            if s in ps:
                return s, size
            s = s.rsplit('_', 1)[0]
            size -= 1
    return np.nan, 0


pattern = re.compile(r'(O|E)')
files = sorted([f for f in listdir(SMPL_PATH)
                if (isfile(join(SMPL_PATH, f)) and pattern.match(f))])

for name in files[:]:
    gc.collect()
    corpus = name.split('.')[0]
    print(corpus)
    
    f = join(SMPL_PATH, f'{corpus}.pickle')
    df = pd.read_pickle(f)
    df = df.reset_index(drop=True)
    df['__2'] = df.token.shift(-1)
    df['__3'] = df.token.shift(-2)
    df['__4'] = df.token.shift(-3)
    df['__5'] = df.token.shift(-4)
    d = df[[TOKEN, '__2', '__3', '__4', '__5']].progress_apply(ngrams, axis=1)
    d = pd.DataFrame.from_records(d.tolist(), columns=['phrase', 'length'])
    mask = ~d.phrase.isnull()
    df = pd.concat([df, d], axis=1).drop(['__2', '__3', '__4', '__5'], axis=1)
    df.loc[mask, TOKEN] = df.loc[mask, 'phrase']
    df.loc[mask, POS] = 'NPHRASE'
    lv = df.length.values
    keep = np.ones_like(lv, dtype=bool)
    length = len(keep)
    for i, v in enumerate(lv):
        if v > 0:
            for j in range(i + 1, min(i + v, length)):
                if lv[j] == 0:
                    keep[j] = False
    df['keep'] = keep
    df = df[df.keep].drop(['phrase', 'length', 'keep'], axis=1)
    f = join(SMPL_PATH, f'{corpus}_wiki_phrases.pickle')
    df.to_pickle(f)
