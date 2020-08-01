import argparse
import gc
import json
import re
from random import shuffle

import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel

from topic_labeling.utils.constants import (
    PHRASES_DIR, POS, TOKEN, HASH, PUNCT, BAD_TOKENS, DATASETS, GOOD_IDS, LDA_DIR, WORD_PATTERN,
    POS_N, POS_NV, POS_NVA, MM_DIR
)
from topic_labeling.utils.utils import init_logging, log_args


def docs_to_lists(token_series):
    return tuple(token_series.tolist())


def texts2corpus(
        documents, tfidf=False, stopwords=None, filter_below=5, filter_above=0.5, keep_n=100000,
        logg=print
):
    logg(f'generating {"tfidf" if tfidf else "bow"} corpus and dictionary')

    dictionary = Dictionary(documents, prune_at=None)
    dictionary.filter_extremes(no_below=filter_below, no_above=filter_above, keep_n=keep_n)

    # filter some noise (e.g. special characters)
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


def make_texts(dataset, nb_files, pos_tags, logg=print):
    sub_dir = 'dewiki' if dataset.startswith('dewi') else 'wiki_phrases'
    dir_path = PHRASES_DIR / sub_dir

    if dataset in {'S', 'speeches'}:
        prefixes = r'^(E|P).*'
    elif dataset in {'F', 'news'}:
        prefixes = r'^(F).*'
    else:
        prefixes = f'({dataset})'
    pattern = re.compile(prefixes)
    files = sorted([f for f in dir_path.iterdir() if f.is_file() and pattern.match(f.name)])

    if not files:
        logg(f'No files found for dataset "{dataset}"')
        exit()

    files = files[:nb_files]
    # pattern explained:
    # - may have leading digits (allows '21_century')
    # - must have at least one character from the German alphabet (removes '1.')
    # - must have at least two alphanumeric characters (allows 'A4', but removes 'C.')

    good_ids = None
    if dataset in GOOD_IDS:
        good_ids = pd.read_pickle(GOOD_IDS[dataset])

    if nb_files is not None:
        logg(f'processing {nb_files} files')

    nb_words = 0
    texts = []
    for filename in files:
        gc.collect()

        logg(f'reading {filename}')
        df = pd.read_pickle(filename)
        logg(f'    initial number of words: {len(df)}')

        # some datasets have already been filtered, so this may not affect the data
        if good_ids is not None:
            df = df[df.hash.isin(good_ids.index)]

        # fixing bad POS tagging
        mask = df.token.isin(list('[]<>/–%{}'))
        df.loc[mask, POS] = PUNCT

        # using only certain POS tags
        if pos_tags:
            df = df[df.POS.isin(pos_tags)]
        df[TOKEN] = df[TOKEN].map(lambda x: x.strip('-/'))
        # TODO: next line probably redundant
        df = df[df.token.str.len() > 1]
        df = df[~df.token.isin(BAD_TOKENS)]
        print(len(df))
        df = df[df.token.str.match(WORD_PATTERN)]
        print(len(df))
        nb_words += len(df)
        logg(f'    remaining number of words: {len(df)}')

        # groupby sorts the documents by hash-id
        # which is equal to shuffling the dataset before building the model
        df = df.groupby([HASH])[TOKEN].agg(docs_to_lists)
        logg(f'    number of documents: {len(df)}')
        texts += df.values.tolist()

    # re-shuffle documents
    if len(files) > 1:
        shuffle(texts)
    if nb_files is not None:
        nb_files = min(nb_files, len(files))

    nb_docs = len(texts)
    logg(f'total number of documents: {nb_docs}')
    logg(f'total number of words: {nb_words}')
    stats = dict(dataset=dataset, pos_set=sorted(pos_tags), nb_docs=nb_docs, nb_words=nb_words)
    return texts, stats, nb_files


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--version", type=str, required=False, default='default')
    parser.add_argument("--nb-files", type=int, required=False, default=None)
    parser.add_argument("--pos_tags", nargs='*', type=str, required=False)

    parser.add_argument('--tfidf', dest='tfidf', action='store_true', required=False)
    parser.add_argument('--no-tfidf', dest='tfidf', action='store_false', required=False)
    parser.set_defaults(tfidf=False)

    args = parser.parse_args()

    args.dataset = DATASETS.get(args.dataset, args.dataset)

    if args.pos_tags is None:
        if args.version == 'all':
            args.pos_tags = []
        elif args.version == 'noun':
            args.pos_tags = POS_N
        elif args.version == 'noun-verb':
            args.pos_tags = POS_NV
        elif args.version == 'noun-verb-adj':
            args.pos_tags = POS_NVA
        else:
            args.pos_tags = POS_N
    args.pos_tags = set(args.pos_tags)

    return args.dataset, args.version, args.nb_files, args.pos_tags, args.tfidf, args


def main():
    dataset, version, nb_files, pos_tags, tfidf, args = parse_args()

    corpus_type = "tfidf" if tfidf else "bow"

    logger = init_logging(
        name=f'MM_{dataset}_{corpus_type}', basic=False, to_stdout=True, to_file=True
    )
    logg = logger.info if logger else print
    log_args(logger, args)

    texts, stats, nb_files = make_texts(dataset, nb_files, pos_tags, logg=logg)

    gc.collect()

    file_name = f'{dataset}{nb_files if nb_files else ""}_{version}'
    directory = MM_DIR / version
    directory.mkdir(exist_ok=True, parents=True)

    # --- saving texts ---
    file_path = directory / f'{file_name}_texts.json'
    logg(f'Saving {file_path}')
    with open(file_path, 'w') as fp:
        json.dump(texts, fp, ensure_ascii=False)

    # --- saving stats ---
    file_path = directory / f'{file_name}_stats.json'
    logg(f'Saving {file_path}')
    with open(file_path, 'w') as fp:
        json.dump(stats, fp)

    # generate and save the dataset as bow or tfidf corpus in Matrix Market format,
    # including dictionary, texts (json) and some stats about corpus size (json)
    corpus, dictionary = texts2corpus(
        texts, tfidf=tfidf, filter_below=5, filter_above=0.5, logg=logg
    )

    file_name += f'_{corpus_type}'
    directory = directory / corpus_type
    directory.mkdir(exist_ok=True, parents=True)

    # --- saving corpus ---
    file_path = directory / f'{file_name}.mm'
    logg(f'Saving {file_path}')
    MmCorpus.serialize(str(file_path), corpus)

    # --- saving dictionary ---
    file_path = directory / f'{file_name}.dict'
    logg(f'Saving {file_path}')
    dictionary.save(str(file_path))


if __name__ == '__main__':
    main()
