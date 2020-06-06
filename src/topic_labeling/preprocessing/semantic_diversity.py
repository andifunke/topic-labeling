import argparse
import gc
import json
import re
from collections import Counter
from itertools import chain
from random import shuffle
from typing import Iterable, List

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel, LsiModel

from topic_labeling.topic_modeling.lda import split_corpus
from topic_labeling.utils.constants import (
    SIMPLE_PATH, POS, TOKEN, HASH, PUNCT, DATASETS, GOOD_IDS, WORD_PATTERN,
    POS_N, POS_NV, POS_NVA, SEMD_PATH
)
from topic_labeling.utils.utils import init_logging, log_args


def lsi(
        corpus, dictionary, model_path, nbs_topics=(300,), use_callbacks=False,
        cache_in_memory=False
):

    model_class = 'LSI_model'
    _split_ = '_split' if use_callbacks else ''

    # # --- logging ---
    logger = init_logging(name=model_class, basic=False, to_stdout=True, to_file=True)
    logg = logger.info

    # corpus = MmCorpus(data_file)
    if cache_in_memory:
        logg("Loading corpus into memory")
        corpus = list(corpus)
    if use_callbacks:
        train, test = split_corpus(corpus)
    else:
        train, test = corpus, []
    logg(f"Size of... train_set={len(train)}, test_set={len(test)}")

    # --- train ---
    top_n = 20
    columns = [f'term{x}' for x in range(top_n)] + [f'weight{x}' for x in range(top_n)]
    for nb_topics in nbs_topics:
        gc.collect()

        logg(f"Running {model_class} with {nb_topics} topics")
        model = LsiModel(corpus=train, num_topics=nb_topics, id2word=dictionary)

        model_dir = model_path.parent
        model_dir.mkdir(exist_ok=True, parents=True)

        # --- save topics ---
        topics = model.show_topics(num_words=top_n, formatted=False)
        topics = [list(chain(*zip(*topic[1]))) for topic in topics]
        topics = pd.DataFrame(topics, columns=columns)
        logg(f"Saving topics to {model_path}.csv")
        topics.to_csv(f'{model_path}.csv')

        # --- save model ---
        logg(f'Saving model to {model_path}')
        model.save(str(model_path))


def docs_to_lists(token_series):
    return token_series.tolist()


def log_transform(corpus, dictionary):
    # take the log
    # sparse_matrix = corpus2csc(corpus)
    # print(sparse_matrix)
    # sparse_matrix.data = np.log(sparse_matrix.data)
    # print(sparse_matrix)

    # calculate entropy
    entropy = Counter()
    for context in corpus:
        for index, value in context:
            corpus_freq = dictionary.dfs[index]
            p_c = value / corpus_freq
            ic = p_c * np.log(p_c)
            # print(index, value, corpus_freq, p_c, ic)
            entropy[index] -= ic

    # calculate transformed value
    log_corpus = [[(i, np.log(v) / entropy[i]) for i, v in context] for context in corpus]

    return log_corpus


def remove_infrequent_words(contexts, min_freq):
    print(f"Filtering words with total frequency < {min_freq}")
    counter = Counter(token for context in contexts for token in context)
    filtered = [
        list(filter(lambda x: counter[x] > min_freq, (token for token in context)))
        for context in contexts
    ]
    print(contexts[0])
    print(filtered[0])
    return filtered


def texts2corpus(
        contexts, tfidf=False, stopwords=None, min_contexts=40, filter_above=1, keep_n=200_000,
        logg=print
):
    logg(f"Generating {'tfidf' if tfidf else 'bow'} corpus and dictionary")

    dictionary = Dictionary(contexts, prune_at=None)
    dictionary.filter_extremes(no_below=min_contexts, no_above=filter_above, keep_n=keep_n)

    # filter some noise (e.g. special characters)
    if stopwords:
        stopword_ids = [dictionary.token2id[token] for token in stopwords]
        dictionary.filter_tokens(bad_ids=stopword_ids, good_ids=None)

    bow_corpus = [dictionary.doc2bow(text) for text in contexts]
    if tfidf:
        tfidf_model = TfidfModel(bow_corpus)
        corpus = tfidf_model[bow_corpus]
    else:
        corpus = bow_corpus

    return corpus, dictionary


def chunks_from_documents(documents: Iterable, window_size: int) -> List:
    contexts = []
    for document in documents:
        if len(document) > window_size:
            chunks = [document[x:x+window_size] for x in range(0, len(document), window_size)]
            contexts += chunks
        else:
            contexts.append(document)

    return contexts


def make_contexts(dataset, nb_files, pos_tags, window_size=1000, logg=print):
    sub_dir = 'dewiki' if dataset.startswith('dewi') else 'wiki_phrases'
    dir_path = SIMPLE_PATH / sub_dir

    if dataset in {'S', 'speeches'}:
        prefixes = r'^(E|P).*'
    elif dataset in {'F', 'news'}:
        prefixes = r'^(F).*'
    else:
        prefixes = f'({dataset})'
    pattern = re.compile(prefixes)
    files = sorted([f for f in dir_path.iterdir() if f.is_file() and pattern.match(f.name)])

    if not files:
        logg(f"No files found for dataset '{dataset}'")
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
        logg(f"Processing {nb_files} files")

    nb_words = 0
    texts = []
    for filename in files:
        gc.collect()

        logg(f"Reading {filename}")
        df = pd.read_pickle(filename)
        logg(f"    initial number of words: {len(df)}")

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

        # df = df[df.token.str.len() > 1]
        # df = df[~df.token.isin(BAD_TOKENS)]

        # TODO: do we want to remove non-work tokens?
        df = df[df.token.str.match(WORD_PATTERN)]
        nb_words += len(df)
        logg(f"    remaining number of words: {len(df)}")

        # groupby sorts the documents by hash-id
        # which is equal to shuffling the dataset before building the model
        df = df.groupby([HASH], sort=True)[TOKEN].agg(docs_to_lists)
        documents = df.values.tolist()
        logg(f"    number of documents: {len(documents)}")
        if window_size > 0:
            contexts = chunks_from_documents(documents, window_size)
        else:
            contexts = documents

        logg(f"    number of contexts: {len(contexts)}")
        texts += contexts

    # re-shuffle documents
    if len(files) > 1:
        shuffle(texts)
    if nb_files is not None:
        nb_files = min(nb_files, len(files))

    nb_docs = len(texts)
    logg(f"Total number of documents: {nb_docs}")
    logg(f"Total number of words: {nb_words}")
    stats = dict(dataset=dataset, pos_set=sorted(pos_tags), nb_docs=nb_docs, nb_words=nb_words)
    return texts, stats, nb_files


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--version', type=str, required=False, default='default')
    parser.add_argument('--nb-files', type=int, required=False, default=None)
    parser.add_argument('--window', type=int, required=False, default=1000)
    parser.add_argument('--min_word_freq', type=int, required=False, default=50)
    parser.add_argument('--min_contexts', type=int, required=False, default=40)
    parser.add_argument('--pos_tags', nargs='*', type=str, required=False)

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

    return args


def main():
    args = parse_args()

    corpus_type = 'tfidf' if args.tfidf else 'entropy'

    logger = init_logging(
        name=f"MM_{args.dataset}_{corpus_type}", basic=False, to_stdout=True, to_file=True
    )
    logg = logger.info if logger else print
    log_args(logger, args)

    contexts, stats, nb_files = make_contexts(
        args.dataset, args.nb_files, args.pos_tags, window_size=args.window, logg=logg
    )

    if args.min_word_freq > 0:
        contexts = remove_infrequent_words(contexts, args.min_word_freq)

    gc.collect()

    file_name = f'{args.dataset}{nb_files if nb_files else ""}_{args.version}'
    directory = SEMD_PATH / args.version
    directory.mkdir(exist_ok=True, parents=True)

    # --- saving texts ---
    file_path = directory / f'{file_name}_texts.json'
    logg(f"Saving {file_path}")
    with open(file_path, 'w') as fp:
        json.dump(contexts, fp, ensure_ascii=False)

    # --- saving stats ---
    file_path = directory / f'{file_name}_stats.json'
    logg(f"Saving {file_path}")
    with open(file_path, 'w') as fp:
        json.dump(stats, fp)

    # generate and save the dataset as bow or tfidf corpus in Matrix Market format,
    # including dictionary, texts (json) and some stats about corpus size (json)
    corpus, dictionary = texts2corpus(
        contexts, tfidf=args.tfidf, stopwords=None, min_contexts=args.min_contexts,
        filter_above=1, logg=logg
    )
    log_corpus = log_transform(corpus, dictionary)

    file_name += f'_{corpus_type}'
    directory = directory / corpus_type
    directory.mkdir(exist_ok=True, parents=True)

    # --- saving corpus ---
    file_path = directory / f'{file_name}.mm'
    logg(f"Saving {file_path}")
    MmCorpus.serialize(str(file_path), corpus)

    # --- saving log-corpus ---
    file_path = directory / f'{file_name}_logs.mm'
    logg(f"Saving {file_path}")
    MmCorpus.serialize(str(file_path), log_corpus)

    # --- saving dictionary ---
    file_path = directory / f'{file_name}.dict'
    logg(f"Saving {file_path}")
    dictionary.save(str(file_path))

    lsi(
        corpus=log_corpus, dictionary=dictionary, model_path=directory / 'LSI', nbs_topics=(300,),
        use_callbacks=False, cache_in_memory=False
    )


if __name__ == '__main__':
    main()
