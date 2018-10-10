# coding: utf-8

import gc
from os import listdir, getpid
from os.path import join, isfile
from time import time
import psutil
process = psutil.Process(getpid())

import numpy as np
import pandas as pd
import re
from options import update_from_args
update_from_args()
from options import CORPUS_PREFIXES
from constants import NLP_PATH, HASH, SENT_IDX, ENT_IDX, ENT_TYPE, NOUN_PHRASE, \
    TEXT, TOKEN, TOK_IDX, POS, ENT_IOB, ETL_PATH, SPACE, SMPL_PATH
from project_logging import log
from tqdm import tqdm
tqdm.pandas()


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


# based on Philipp Grawes approach on extracting and normalizing street names
STREET_NAME_LIST = [r'strasse$', r'straße$', r'str$', r'str.$', r'platz', r'gasse$',
                    r'allee$', r'ufer$', r'weg$']
STREET_NAMES = re.compile(r'(' + '|'.join(STREET_NAME_LIST) + ')', re.IGNORECASE)
STREET_PATTERN = re.compile(r"str(\.|a(ss|ß)e)?\b", re.IGNORECASE)
SPECIAL_CHAR = re.compile(r'[^\w&/]+')


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


def main(corpus):

    t0 = time()

    fpath = join(NLP_PATH, corpus + '_nlp.pickle')
    log("reading from " + fpath)
    df = pd.read_pickle(fpath)
    log(memstr())

    if corpus.startswith('dewac'):
        goodids = pd.read_pickle(join(ETL_PATH, 'dewac_good_ids.pickle'))
        df = df[df.hash.isin(goodids.index)]
        df = (
            df
            .groupby(HASH, sort=False, as_index=False)
            .progress_apply(remove_title)
            .reset_index(level=0, drop=True)
        )

    log("extracting spacy NER")
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
    log(memstr())

    log("extracting spacy noun chunks")
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
    log(memstr())

    log("intersecting both extraction methods")
    df_phrases = df_ent.append(df_np)
    del df_ent, df_np
    gc.collect()

    mask = df_phrases.duplicated([HASH, SENT_IDX, TOK_IDX])
    df_phrases = df_phrases[mask]
    # set column token-index to start of phrase and add column column for the token-indexes instead
    df_phrases['tok_set'] = df_phrases[TOK_IDX]
    df_phrases[TOK_IDX] = df_phrases[TOK_IDX].apply(lambda x: x[0])
    log(memstr())

    log("extracting streets")
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
    log(memstr())

    log("insert phrases to original tokens")
    df_glued = insert_phrases(df, df_phrases)
    # log('df_glued: Memory consumed: {:.2f} Mb'.format(df_glued.memory_usage(index=True, deep=False)))
    # log('df_glued: Memory consumed (deep): {:.2f} Mb'.format(df_glued.memory_usage(index=True, deep=True)))
    del df_phrases
    log("insert locations / streets")
    df_glued = insert_phrases(df_glued, df_loc)
    del df_loc
    gc.collect()

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
    log(memstr())

    # log(h.heap())
    write_path = join(SMPL_PATH, corpus + '_simple.pickle')
    gc.collect()

    log("writing to " + write_path)
    df_glued.to_pickle(write_path)

    t1 = int(time() - t0)
    log("done in {:02d}:{:02d}:{:02d}".format(t1//3600, (t1//60) % 60, t1 % 60))


if __name__ == "__main__":
    t0 = time()

    ### --- run ---
    log("##### START #####")

    # filter files for certain prefixes
    prefixes = r'^(' + '|'.join(CORPUS_PREFIXES) + r').'
    pattern = re.compile(prefixes)
    files = sorted([f for f in listdir(NLP_PATH)
                    if (isfile(join(NLP_PATH, f)) and pattern.match(f))])

    for name in files:
        corpus_name = name.split('_nlp.')[0]
        main(corpus_name)
        log(memstr())

    t1 = int(time() - t0)
    log("all done in {:02d}:{:02d}:{:02d}".format(t1//3600, (t1//60) % 60, t1 % 60))
