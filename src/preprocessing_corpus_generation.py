import re
from os import listdir, makedirs
from os.path import join, isfile, exists
from random import shuffle
import json
import argparse
import gc

import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel

from utils import init_logging, log_args
from constants import (
    SMPL_PATH, POS, NOUN, PROPN, TOKEN, HASH, PUNCT, BAD_TOKENS, DATASETS,
    GOOD_IDS, NER, NPHRASE, VERB, ADJ, ADV, LDA_PATH,
    NOUN_PATTERN, POS_N, POS_NV, POS_NVA)


def docs_to_lists(token_series):
    return tuple(token_series.tolist())


def texts2corpus(
        documents, tfidf=False, stopwords=None, filter_below=5, filter_above=0.5, keep_n=100000,
        logg=print
):
    logg(f'generating {"tfidf" if tfidf else "bow"} corpus and dictionary')

    dictionary = Dictionary(documents, prune_at=None)
    dictionary.filter_extremes(no_below=filter_below, no_above=filter_above, keep_n=keep_n)

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


def make_texts(dataset, nbfiles, pos_tags, logg=print):
    sub_dir = 'dewiki' if dataset.startswith('dewi') else 'wiki_phrases'
    dir_path = join(SMPL_PATH, sub_dir)

    if dataset in {'S', 'speeches'}:
        prefixes = r'^(E|P).*'
    elif dataset in {'F', 'news'}:
        prefixes = r'^(F).*'
    else:
        prefixes = f'({dataset})'
    pattern = re.compile(prefixes)
    files = sorted([f for f in listdir(dir_path)
                    if (isfile(join(dir_path, f)) and pattern.match(f))])

    files = files[:nbfiles]
    # pattern explained:
    # - may have leading digits (allows '21_century')
    # - must have at least one character from the German alphabet (renoves '1.')
    # - must have at least two alphanumeric characters (allows 'A4', but removes 'C.')

    goodids = None
    if dataset in GOOD_IDS:
        goodids = pd.read_pickle(GOOD_IDS[dataset])

    if nbfiles is not None:
        logg(f'processing {nbfiles} files')

    nb_words = 0
    texts = []
    for filename in files:
        gc.collect()
        full_path = join(dir_path, filename)
        if not isfile(full_path):
            continue

        logg(f'reading {filename}')
        df = pd.read_pickle(join(dir_path, filename))
        logg(f'    initial number of words: {len(df)}')
        if goodids is not None:
            # some datasets have already been filtered so you may not see a difference in any case
            df = df[df.hash.isin(goodids.index)]

        # fixing bad POS tagging
        mask = df.token.isin(list('[]<>/–%{}'))
        df.loc[mask, POS] = PUNCT

        # using only certain POS tags
        df = df[df.POS.isin(pos_tags)]
        df[TOKEN] = df[TOKEN].map(lambda x: x.strip('-/'))
        # TODO: next line probably redundant
        df = df[df.token.str.len() > 1]
        df = df[~df.token.isin(BAD_TOKENS)]
        print(len(df))
        df = df[df.token.str.match(NOUN_PATTERN)]
        print(len(df))
        nb_words += len(df)
        logg(f'    remaining number of words: {len(df)}')

        # groupby sorts the documents by hash-id
        # which is equal to shuffeling the dataset before building the model
        df = df.groupby([HASH])[TOKEN].agg(docs_to_lists)
        logg(f'    number of documents: {len(df)}')
        texts += df.values.tolist()

    # re-shuffle documents
    if len(files) > 1:
        shuffle(texts)
    if nbfiles is not None:
        nbfiles = min(nbfiles, len(files))

    nb_docs = len(texts)
    logg(f'total number of documents: {nb_docs}')
    logg(f'total number of words: {nb_words}')
    stats = dict(dataset=dataset, pos_set=sorted(pos_tags), nb_docs=nb_docs, nb_words=nb_words)
    return texts, stats, nbfiles


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--version", type=str, required=False, default='default')
    parser.add_argument("--nbfiles", type=int, required=False, default=None)
    parser.add_argument("--pos_tags", nargs='*', type=str, required=False)

    parser.add_argument('--tfidf', dest='tfidf', action='store_true', required=False)
    parser.add_argument('--no-tfidf', dest='tfidf', action='store_false', required=False)
    parser.set_defaults(tfidf=False)

    args = parser.parse_args()

    args.dataset = DATASETS.get(args.dataset, args.dataset)

    if args.pos_tags is None:
        if args.version == 'noun':
            args.pos_tags = POS_N
        elif args.version == 'noun-verb':
            args.pos_tags = POS_NV
        elif args.version == 'noun-verb-adj':
            args.pos_tags = POS_NVA
        else:
            args.pos_tags = POS_N
    args.pos_tags = set(args.pos_tags)

    return args.dataset, args.version, args.nbfiles, args.pos_tags, args.tfidf, args


def main():
    dataset, version, nbfiles, pos_tags, tfidf, args = parse_args()

    corpus_type = "tfidf" if tfidf else "bow"

    logger = init_logging(name=f'MM_{dataset}_{corpus_type}', basic=False, to_stdout=True, to_file=True)
    logg = logger.info if logger else print
    log_args(logger, args)

    texts, stats, nbfiles = make_texts(dataset, nbfiles, pos_tags, logg=logg)
    gc.collect()

    file_name = f'{dataset}{nbfiles if nbfiles else ""}_{version}'
    directory = join(LDA_PATH, version)
    if not exists(directory):
        makedirs(directory)

    # --- saving texts ---
    file_path = join(directory, f'{file_name}_texts.json')
    logg(f'Saving {file_path}')
    with open(file_path, 'w') as fp:
        json.dump(texts, fp, ensure_ascii=False)

    # --- saving stats ---
    file_path = join(directory, f'{file_name}_stats.json')
    logg(f'Saving {file_path}')
    with open(file_path, 'w') as fp:
        json.dump(stats, fp)

    # generate and save the dataset as bow or tfidf corpus in Matrix Market format,
    # including dictionary, texts (json) and some stats about corpus size (json)
    corpus, dictionary = texts2corpus(texts, tfidf=tfidf, filter_below=5, filter_above=0.5, logg=logg)

    file_name += f'_{corpus_type}'
    directory = join(directory, corpus_type)

    # --- saving corpus ---
    file_path = join(directory, f'{file_name}.mm')
    logg(f'Saving {file_path}')
    MmCorpus.serialize(file_path, corpus)

    # --- saving dictionary ---
    file_path = join(directory, f'{file_name}.dict')
    logg(f'Saving {file_path}')
    dictionary.save(file_path)


if __name__ == '__main__':
    main()
