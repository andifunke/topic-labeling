# coding: utf-8

"""
This version of phrase_extraction applies the process to smaller batches in order to save memory.
"""

import gc
from os import listdir, getpid
from os.path import join, isfile
from time import time
import psutil

from utils import init_logging

process = psutil.Process(getpid())
import numpy as np
import pandas as pd
import re
from constants import (
    NLP_PATH, HASH, SENT_IDX, ENT_IDX, ENT_TYPE, NOUN_PHRASE, TEXT, TOKEN, TOK_IDX, POS, ENT_IOB,
    ETL_PATH, SPACE, SMPL_PATH, BAD_FIRST_PHRASE_TOKEN, PUNCT
)
from tqdm import tqdm
tqdm.pandas()


# based on Philipp Grawes approach on extracting and normalizing street names
STREET_NAME_LIST = [
    r'strasse$', r'straße$', r'str$', r'str.$', r'platz', r'gasse$', r'allee$', r'ufer$', r'weg$'
]
STREET_NAMES = re.compile(r'(' + '|'.join(STREET_NAME_LIST) + ')', re.IGNORECASE)
STREET_PATTERN = re.compile(r"str(\.|a(ss|ß)e)?\b", re.IGNORECASE)
SPECIAL_CHAR = re.compile(r'[^\w&/]+')
GOOD_IDS_DEWAC = None
GOOD_IDS_DEWIKI = None
PS = None
LOG_FUNC = print


def logg(msg):
    LOG_FUNC(msg)


def memstr():
    rss = "RSS: {:.2f} GB".format(process.memory_info().rss / (2**30))
    vms = "VMS: {:.2f} GB".format(process.memory_info().vms / (2**30))
    return rss + ' | ' + vms


def concat_entities(column):
    if column.name in {HASH, SENT_IDX, ENT_IDX, ENT_TYPE}:
        return column.values[0]
    if column.name in {'tok_idx', NOUN_PHRASE, 'np'}:
        return tuple(column.values)
    if column.name in {TEXT, TOKEN}:
        return column.str.cat(sep='_')
    return False


def get_removable_tokens(df_in):
    remove_token = []
    for i, sent_idx, tok_set in df_in.itertuples():
        for tok_idx in tok_set:
            remove_token.append((sent_idx, tok_idx))
    df_out = (
        pd.DataFrame
        .from_records(remove_token, columns=[SENT_IDX, TOK_IDX])
        .assign(hash=0, ent_idx=0)
    )
    return df_out


def insert_phrases(df_orig, df_insert):
    """add phrases and replace overlapping tokens"""
    # df_removable_tokens: this DataFrame contains all token-idx we want to replace with phrases
    df_removable_tokens = get_removable_tokens(df_insert[[SENT_IDX, 'tok_set']])
    df_combined = (
        df_orig
        # remove original unigram tokens
        .append(df_removable_tokens)
        .drop_duplicates(subset=[SENT_IDX, TOK_IDX], keep=False)
        .dropna(subset=[TOKEN])
        # insert concatenated phrase tokens
        .append(df_insert)
        .sort_values([SENT_IDX, TOK_IDX])
    )
    return df_combined


def aggregate_streets(column):
    if column.name in {HASH, SENT_IDX, ENT_IDX}:
        return column.values[0]
    if column.name in {'tok_idx', NOUN_PHRASE, 'np'}:
        return tuple(column.values)
    if column.name == TEXT:
        return column.str.cat(sep='_')
    if column.name == TOKEN:
        street_candidate = False
        for k, token in column.iteritems():
            if re.search(STREET_NAMES, token):
                street_candidate = True
        if street_candidate:
            if len(column) == 1 and re.fullmatch(STREET_NAMES, column.values[0]):
                return False
            else:
                street_name = column.str.cat(sep=' ')
                street_name = STREET_PATTERN.sub('straße', street_name)
                street_name = SPECIAL_CHAR.sub('_', street_name)
                street_name = street_name.strip('_').title()
                return street_name
    return False


def remove_title(x):
    """ apply this function only on the dewac corpus.
     It removes the rows up to the first line feed (inclusive), i.e. the document title. """
    ln_argmx = (x.text.values == '\n').argmax()
    return x[ln_argmx+1:]


def preprocess_dewac(df):
    global GOOD_IDS_DEWAC
    if GOOD_IDS_DEWAC is None:
        GOOD_IDS_DEWAC = pd.read_pickle(join(ETL_PATH, 'dewac_good_ids.pickle'))
    df = df[df.hash.isin(GOOD_IDS_DEWAC.index)]
    df = (
        df
        .groupby(HASH, sort=False, as_index=False)
        .progress_apply(remove_title)
        .reset_index(level=0, drop=True)
    )
    return df


def preprocess_dewiki(df):
    global GOOD_IDS_DEWIKI
    if GOOD_IDS_DEWIKI is None:
        GOOD_IDS_DEWIKI = pd.read_pickle(join(ETL_PATH, 'dewiki_good_ids.pickle'))
    df = df[df.hash.isin(GOOD_IDS_DEWIKI.index)]
    return df


def ngrams(ser):
    if ser[0].lower() not in BAD_FIRST_PHRASE_TOKEN:
        s = ser.str.cat(sep='_')
        size = len(ser)
        while size > 1:
            if s in PS:
                return s, size
            s = s.rsplit('_', 1)[0]
            size -= 1
    return np.nan, 0


def insert_wikipedia_phrases(df):
    # TODO: fix missing concatenation for TEXT
    # TODO: ignore n-grams beyond sentence segments
    global PS
    if PS is None:
        p = pd.read_pickle(join(ETL_PATH, 'dewiki_phrases_lemmatized.pickle'))
        p = p[p.title_len > 1]
        PS = set(p.text.append(p.token))

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
    return df


def process_subset(df):
    logg("extracting spacy NER")
    # phrases have an ent-index > 0 and we don't care about whitespace
    df_ent = df.query('ent_idx > 0 & POS != "SPACE"')
    df_ent = (
        df_ent[df_ent.groupby(ENT_IDX).ent_idx.transform(len) > 1]
        .groupby(ENT_IDX, as_index=False).agg(concat_entities)  # concatenate entities
        .assign(
            # add the number of tokens per entity as a new column
            length=lambda x: x.tok_idx.apply(lambda y: len(y)),
            POS='NER',  # annotations
            ent_iob='P',
        )
        .astype({  # set annoation columns as categorical for memory savings
            POS: "category",
            ENT_IOB: "category",
            ENT_TYPE: "category"
        })
    )
    logg('collect: %d' % gc.collect())
    logg(memstr())

    logg("extracting spacy noun chunks")
    df_np = df.query('noun_phrase > 0 & POS not in ["SPACE", "NUM", "DET", "SYM"]')
    df_np = (
        df_np[df_np.groupby(NOUN_PHRASE).noun_phrase.transform(len) > 1]
        .groupby(NOUN_PHRASE, as_index=False).agg(concat_entities)
        .assign(
            length=lambda x: x.tok_idx.apply(lambda y: len(y)),
            POS='NPHRASE',
            ent_iob='P',
        )
        .astype({
            POS: "category",
            ENT_IOB: "category",
            ENT_TYPE: "category"
        })
    )
    logg('collect: %d' % gc.collect())
    logg(memstr())

    logg("intersecting both extraction methods")
    df_phrases = df_ent.append(df_np)
    del df_ent, df_np
    logg('collect: %d' % gc.collect())
    logg(memstr())

    mask = df_phrases.duplicated([HASH, SENT_IDX, TOK_IDX])
    df_phrases = df_phrases[mask]
    # set column token-index to start of phrase and add column column for the token-indexes instead
    df_phrases['tok_set'] = df_phrases[TOK_IDX]
    df_phrases[TOK_IDX] = df_phrases[TOK_IDX].apply(lambda x: x[0])

    logg("insert phrases to original tokens")
    df_glued = insert_phrases(df, df_phrases)
    del df_phrases

    logg("extracting streets")
    df_loc = (
        df
        .loc[(df[ENT_IDX] > 0) & (df.POS != SPACE)]
        .groupby(ENT_IDX, as_index=False).agg(aggregate_streets)
        .query('token != False')
        .assign(
            length=lambda x: x.tok_idx.apply(lambda y: len(y)),
            tok_set=lambda x: x.tok_idx,
            tok_idx=lambda x: x.tok_idx.apply(lambda y: y[0]),
            POS='PROPN',
            ent_iob='L',
            ent_type='STREET'
        )
        .astype({
            POS: "category",
            ENT_IOB: "category",
            ENT_TYPE: "category"
        })
    )
    del df
    logg('collect: %d' % gc.collect())
    logg(memstr())

    logg("insert locations / streets")
    df_glued = insert_phrases(df_glued, df_loc)
    del df_loc
    logg('collect: %d' % gc.collect())
    logg(memstr())

    # simplify dataframe and store
    df_glued = (
        df_glued
        .loc[df_glued.POS != 'SPACE', [HASH, POS, SENT_IDX, TOK_IDX, TOKEN]]
        .astype({
            HASH: np.int64,
            POS: "category",
            SENT_IDX: np.int32,
            TOK_IDX: np.int32,
        })
    )
    logg('collect: %d' % gc.collect())
    logg(memstr())

    logg("insert wikipedia phrases")
    df_glued = insert_wikipedia_phrases(df_glued)
    logg('collect: %d' % gc.collect())
    logg(memstr())
    return df_glued


def main(corpus, batch_size=None):
    t0 = time()

    fpath = join(NLP_PATH, corpus + '_nlp.pickle')
    logg("reading from " + fpath)
    df_main = pd.read_pickle(fpath)
    logg(memstr())

    if corpus.startswith('dewac'):
        df_main = preprocess_dewac(df_main)
    elif corpus.startswith('dewiki'):
        df_main = preprocess_dewiki(df_main)

    # fixes wrong POS tagging for punctuation
    mask_punct = df_main[TOKEN].isin(list('[]<>/–%{}'))
    df_main.loc[mask_punct, POS] = PUNCT

    df_main = df_main.groupby(HASH, sort=False)
    length = len(df_main)

    # init
    t_a = time()
    groups_tmp = []
    last_cnt = 1
    for i, grp in enumerate(df_main, last_cnt):
        groups_tmp.append(grp[1])
        # process and save in batches of size batch_size if batch_size is not None
        # or if the last document is reached
        if (batch_size is not None and (i % batch_size == 0)) or (i == length):
            logg('process {:d}:{:d}'.format(last_cnt, i))
            df_glued = process_subset(pd.concat(groups_tmp))
            write_path = join(SMPL_PATH, corpus + '__{:d}_simple.pickle'.format(i))
            logg(memstr())
            logg('collect: %d' % gc.collect())
            logg(memstr())

            logg("writing to " + write_path)
            df_glued.to_pickle(write_path)
            t_b = int(time() - t_a)
            logg("subset done in {:02d}:{:02d}:{:02d}".format(t_b//3600, (t_b//60) % 60, t_b % 60))

            # reset
            t_a = time()
            groups_tmp = []
            last_cnt = i+1

    t1 = int(time() - t0)
    logg("done in {:02d}:{:02d}:{:02d}".format(t1//3600, (t1//60) % 60, t1 % 60))


if __name__ == "__main__":
    from options import update_from_args
    update_from_args()
    from options import CORPUS_PREFIXES
    logger = init_logging('Phrase_extraction')
    LOG_FUNC = logger.info

    t_0 = time()

    # filter files for certain prefixes
    prefixes = r'^(' + '|'.join(CORPUS_PREFIXES) + r').'
    pattern = re.compile(prefixes)
    files = sorted([
        f for f in listdir(NLP_PATH)
        if (isfile(join(NLP_PATH, f)) and pattern.match(f))
    ])

    for name in files:
        corpus_name = name.split('_nlp.')[0]
        main(corpus_name, batch_size=10000)

    t_1 = int(time() - t_0)
    logg("all done in {:02d}:{:02d}:{:02d}".format(t_1//3600, (t_1//60) % 60, t_1 % 60))
