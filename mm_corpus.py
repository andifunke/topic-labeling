from os import listdir, makedirs
from os.path import join, isfile, exists
from random import shuffle
import json
import argparse
import gc

import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel

from train_w2v import init_logging
from constants import (
    ETL_PATH, SMPL_PATH, POS, NOUN, PROPN, TOKEN, HASH, PUNCT, BAD_TOKENS, DATASETS,
    GOOD_IDS)


def docs_to_lists(token_series):
    return tuple(token_series.tolist())


def texts2corpus(documents, tfidf=True, stopwords=None, filter_below=5, filter_above=0.5):
    dictionary = Dictionary(documents)
    dictionary.filter_extremes(no_below=filter_below, no_above=filter_above)

    # filter some noice (e.g. special characters)
    if stopwords:
        stopword_ids = [dictionary.token2id[token] for token in stopwords]
        dictionary.filter_tokens(bad_ids=stopword_ids, good_ids=None)

    bow_corpus = [dictionary.doc2bow(text) for text in documents]
    if tfidf:
        tfidf_model = TfidfModel(bow_corpus)
        corpus = tfidf_model[bow_corpus]
    else:
        corpus = bow_corpus

    return corpus, dictionary


def make_texts(dataset, nbfiles, pos_tags, logger):
    sub_dir = 'dewiki' if dataset.startswith('dewi') else 'wiki_phrases'
    dir_path = join(SMPL_PATH, sub_dir)
    files = sorted([f for f in listdir(dir_path) if f.startswith(dataset)])
    files = files[:nbfiles]

    goodids = None
    if dataset in GOOD_IDS:
        goodids = pd.read_pickle(GOOD_IDS[dataset])

    if nbfiles is not None:
        logger.info(f'processing {nbfiles} files')

    nb_words = 0
    texts = []
    for filename in files:
        gc.collect()
        full_path = join(dir_path, filename)
        if not isfile(full_path):
            continue

        logger.info(f'reading {filename}')
        df = pd.read_pickle(join(dir_path, filename))
        logger.info(f'    initial number of words: {len(df)}')
        if goodids is not None:
            # some datasets have already been filtered so you may not see a difference in any case
            df = df[df.hash.isin(goodids.index)]

        # fixing bad POS tagging
        mask = df.token.isin(list('[]<>/–%'))
        df.loc[mask, POS] = PUNCT

        # using only certain POS tags
        df = df[df.POS.isin(pos_tags)]
        df[TOKEN] = df[TOKEN].map(lambda x: x.strip('-/'))
        df = df[df.token.str.len() > 1]
        df = df[~df.token.isin(BAD_TOKENS)]
        nb_words += len(df)
        logger.info(f'    remaining number of words: {len(df)}')

        # groupby sorts the documents by hash-id
        # which is equal to shuffeling the dataset before building the model
        df = df.groupby([HASH])[TOKEN].agg(docs_to_lists)
        logger.info(f'    number of documents: {len(df)}')
        texts += df.values.tolist()

    # re-shuffle documents
    if len(files) > 1:
        shuffle(texts)
    if nbfiles is not None:
        nbfiles = min(nbfiles, len(files))

    nb_docs = len(texts)
    logger.info(f'total number of documents: {nb_docs}')
    logger.info(f'total number of words: {nb_words}')
    stats = dict(dataset=dataset, pos_set=sorted(pos_tags), nb_docs=nb_docs, nb_words=nb_words)
    return texts, stats, nbfiles


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--nbfiles", type=int, required=False, default=None)
    parser.add_argument("--pos_tags", nargs='*', type=str, required=False,
                        default=[NOUN, PROPN, 'NER', 'NPHRASE'])
    parser.add_argument('--use_tfidf', dest='use_tfidf', action='store_true', required=False)
    parser.add_argument('--no-use_tfidf', dest='use_tfidf', action='store_false', required=False)
    parser.set_defaults(use_tfidf=False)
    args = parser.parse_args()

    dataset = DATASETS.get(args.dataset, args.dataset)
    return dataset, args.version, args.nbfiles, set(args.pos_tags), args.use_tfidf


def main():
    dataset, version, nbfiles, pos_tags, use_tfidf = parse_args()

    logger = init_logging(
        name=dataset, basic=False, to_stdout=True, to_file=True, log_file=f'MM_{dataset}.log'
    )
    logger.info(dataset)
    logger.info(f'version {version}')
    logger.info(f'nbfiles {nbfiles}')
    logger.info(f'use_tfidf {use_tfidf}')
    logger.info(pos_tags)

    texts, stats, nbfiles = make_texts(dataset, nbfiles, pos_tags, logger)
    gc.collect()

    # generate and save the dataset as bow or tfidf corpus in Matrix Market format,
    # including dictionary, texts (json) and some stats about corpus size (json)
    corpus, dictionary = texts2corpus(texts, tfidf=use_tfidf, filter_below=5, filter_above=0.5)

    # --- saving data ---
    directory = join(ETL_PATH, 'LDAmodel', version)
    if not exists(directory):
        makedirs(directory)

    corpus_str = 'tfidf' if use_tfidf else 'bow'
    nbfiles_str = f'_nbfiles{nbfiles:02d}' if nbfiles else ''
    file_name = f'{dataset}{nbfiles_str}_{version}_{corpus_str}'

    file_path = join(directory, f'{file_name}.mm')
    logger.info(f'Saving {file_path}')
    MmCorpus.serialize(file_path, corpus)

    file_path = join(directory, f'{file_name}.dict')
    logger.info(f'Saving {file_path}')
    dictionary.save(file_path)

    file_path = join(directory, f'{file_name}_texts.json')
    logger.info(f'Saving {file_path}')
    with open(file_path, 'w') as fp:
        json.dump(texts, fp, ensure_ascii=False)

    file_path = join(directory, f'{file_name}_stats.json')
    logger.info(f'Saving {file_path}')
    with open(file_path, 'w') as fp:
        json.dump(stats, fp)


if __name__ == '__main__':
    main()
