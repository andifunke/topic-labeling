import argparse
import gc
import json
import re
from collections import Counter
from pathlib import Path
from random import shuffle
from time import time
from typing import Iterable, List

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.matutils import corpus2dense, corpus2csc
from gensim.models import TfidfModel, LsiModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from topic_labeling.topic_modeling.lda import split_corpus
from topic_labeling.utils.constants import (
    SIMPLE_PATH, POS, TOKEN, HASH, PUNCT, DATASETS, POS_N, POS_NV, POS_NVA, SEMD_PATH
)


tqdm.pandas()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--version', type=str, required=False, default='default')
    parser.add_argument('--nb-files', type=int, required=False, default=None)
    parser.add_argument('--window', type=int, required=False, default=1000)
    parser.add_argument('--min-word-freq', type=int, required=False, default=50)
    parser.add_argument('--min-contexts', type=int, required=False, default=40)
    parser.add_argument('--nb-topics', type=int, required=False, default=300)
    parser.add_argument('--pos-tags', nargs='*', type=str, required=False)
    parser.add_argument('--terms', type=str, required=False, help="File path containing terms")

    parser.add_argument(
        '--tfidf', dest='tfidf', action='store_true', required=False
    )
    parser.add_argument(
        '--no-tfidf', dest='tfidf', action='store_false', required=False
    )
    parser.set_defaults(tfidf=False)

    parser.add_argument(
        '--make-contexts', dest='make_contexts', action='store_true', required=False
    )
    parser.add_argument(
        '--load-contexts', dest='make_contexts', action='store_false', required=False
    )
    parser.set_defaults(make_contexts=True)

    parser.add_argument(
        '--make-corpus', dest='make_corpus', action='store_true', required=False
    )
    parser.add_argument(
        '--load-corpus', dest='make_corpus', action='store_false', required=False
    )
    parser.set_defaults(make_corpus=True)

    parser.add_argument(
        '--make-lsi', dest='make_lsi', action='store_true', required=False
    )
    parser.add_argument(
        '--load-lsi', dest='make_lsi', action='store_false', required=False
    )
    parser.set_defaults(make_lsi=True)

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


def calculate_semantic_diversity(terms, dictionary, corpus, document_vectors):
    csc_matrix = corpus2csc(corpus, dtype=np.float32)

    semd_values = {}
    for term in tqdm(terms, total=len(terms)):
        try:
            term_id = dictionary.token2id[term]
            term_docs_sparse = csc_matrix.getrow(term_id)
            term_doc_ids = term_docs_sparse.nonzero()[1]
            term_doc_vectors = document_vectors[term_doc_ids]
            similarities = cosine_similarity(term_doc_vectors)
            avg_similarity = np.mean(similarities)
            semd = -np.log10(avg_similarity)
            semd_values[term] = semd
        except KeyError:
            semd_values[term] = np.nan

    return pd.Series(semd_values)


def lsi_transform(corpus, dictionary, nb_topics=300, use_callbacks=False, cache_in_memory=False):
    if cache_in_memory:
        print("Loading corpus into memory")
        corpus = list(corpus)
    if use_callbacks:
        train, test = split_corpus(corpus)
    else:
        train, test = corpus, []
    print(f"Size of... train_set={len(train)}, test_set={len(test)}")

    # --- train ---
    print(f"Training LSI model with {nb_topics} topics")
    model = LsiModel(corpus=train, num_topics=nb_topics, id2word=dictionary, dtype=np.float32)

    # --- get vectors ---
    term_vectors = model.projection.u
    term_vectors = pd.DataFrame(term_vectors, index=dictionary.token2id.keys())

    lsi_corpus = model[corpus]
    document_vectors = corpus2dense(lsi_corpus, 300, num_docs=len(corpus)).T
    document_vectors = pd.DataFrame(document_vectors)

    return model, document_vectors, term_vectors


def docs_to_lists(token_series):
    return token_series.tolist()


def entropy_transform(corpus, dictionary, epsilon=.0001):
    # calculate "entropy" per token
    entropy = Counter()
    for context in corpus:
        for index, value in context:
            corpus_freq = dictionary.dfs[index]
            p_c = value / corpus_freq
            ic = -np.log(p_c)
            # print(index, value, corpus_freq, p_c, ic)
            entropy[index] += p_c * ic
            if entropy[index] == 0:
                raise ValueError(
                    f"Entropy calculated as 0\n"
                    f"{index}, {value}, {dictionary.id2token[index]}"
                )

    # calculate transformed value
    entropy_corpus = [
        [(i, (np.log(v) + epsilon) / entropy[i]) for i, v in context]
        for context in corpus
    ]

    return entropy_corpus


def tfidf_transform(bow_corpus):
    tfidf_model = TfidfModel(bow_corpus)
    tfidf_corpus = tfidf_model[bow_corpus]
    return tfidf_corpus


def calculate_chunks(df, window):
    print(f"Calculating chunks")
    t0 = time()

    def _chunk(group):
        length = len(group)
        if length <= window:
            group['chunk'] = 1
        else:
            index = np.arange(length)
            index = index % window == 0
            index = index.cumsum()
            group['chunk'] = index
        return group

    df = df.groupby(HASH, sort=False, as_index=False).progress_apply(_chunk)
    print(f'time: {time() - t0}')
    return df


def remove_infrequent_words(df, min_freq, min_contexts):
    print(f"Filtering words with total frequency < {min_freq}.")
    size = len(df)
    frequencies = df[TOKEN].value_counts()
    frequencies = frequencies[frequencies >= min_freq]
    df = df[df[TOKEN].isin(frequencies.index)]
    print(f"{size} => {len(df)} - {size - len(df)} words removed.")

    print(f"Filtering words appearing in less than {min_contexts} contexts.")
    size = len(df)
    frequencies = df.groupby([HASH, 'chunk'])[TOKEN].unique().explode().value_counts()
    frequencies = frequencies[frequencies >= min_contexts]
    df = df[df[TOKEN].isin(frequencies.index)]
    print(f"{size} => {len(df)} - {size - len(df)} words removed.")

    return df


def texts2corpus(contexts, stopwords=None, filter_above=1, keep_n=200_000):
    print(f"Generating bow corpus and dictionary")

    dictionary = Dictionary(contexts, prune_at=None)
    dictionary.filter_extremes(no_above=filter_above, keep_n=keep_n)

    # filter some noise (e.g. special characters)
    if stopwords:
        stopword_ids = [dictionary.token2id[token] for token in stopwords]
        dictionary.filter_tokens(bad_ids=stopword_ids, good_ids=None)

    bow_corpus = [dictionary.doc2bow(text) for text in contexts]

    return bow_corpus, dictionary


# TODO: deprecated
def chunks_from_documents(documents: Iterable, window_size: int) -> List:
    contexts = []
    for document in documents:
        if len(document) > window_size:
            chunks = [document[x:x+window_size] for x in range(0, len(document), window_size)]
            contexts += chunks
        else:
            contexts.append(document)

    return contexts


def make_contexts(
        dataset, min_freq=0, min_contexts=0, nb_files=None, pos_tags=None, window_size=1000
):
    sub_dir = 'dewiki' if dataset.startswith('dewiki') else 'wiki_phrases'
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
        print(f"No files found for dataset '{dataset}'")
        raise SystemExit

    if nb_files is not None:
        files = files[:nb_files]
        print(f"Processing {nb_files} files")

    nb_words = 0
    texts = []
    for filename in files:
        gc.collect()

        print(f"Reading {filename}")
        df = pd.read_pickle(filename)
        df = df.drop(['sent_idx', 'tok_idx'], axis=1)

        # filtering POS tags
        if pos_tags:
            print(f"    initial number of words: {len(df)}")
            # fixing wrong POS tagging
            # TODO: deprecated
            mask = df.token.isin(list('[]<>/–%{}'))
            df.loc[mask, POS] = PUNCT
            df = df[df.POS.isin(pos_tags)]
            print(f"    remaining number of words: {len(df)}")
        df = df.drop(POS, axis=1)
        # TODO: deprecated - migrate to earlier pipeline
        df[TOKEN] = df[TOKEN].map(lambda x: x.strip('-/'))

        df = calculate_chunks(df, window_size)

        # TODO: if read from multiple files this needs to be generalized
        df = remove_infrequent_words(df, min_freq, min_contexts)
        nb_words += len(df)

        # - shuffling -
        nb_documents = len(df[HASH].unique())
        print(f"    number of documents: {nb_documents}")
        contexts = df.groupby([HASH, 'chunk'], sort=True)[TOKEN].agg(docs_to_lists).to_list()
        print(f"    number of contexts: {len(contexts)}")
        texts += contexts

    # re-shuffle documents
    if len(files) > 1:
        shuffle(texts)
    if nb_files is not None:
        nb_files = min(nb_files, len(files))

    nb_contexts = len(texts)
    print(f"Total number of contexts: {nb_contexts}")
    print(f"Total number of words: {nb_words}")
    stats = dict(dataset=dataset, pos_set=sorted(pos_tags), nb_docs=nb_contexts, nb_words=nb_words)
    return texts, stats, nb_files


def get_contexts(args, directory, file_name):
    if args.make_contexts:
        # - make contexts -
        contexts, stats, nb_files = make_contexts(
            dataset=args.dataset,
            min_freq=args.min_word_freq,
            min_contexts=args.min_contexts,
            nb_files=args.nb_files,
            pos_tags=args.pos_tags,
            window_size=args.window,
        )
        gc.collect()

        # - save contexts -
        file_path = directory / f'{file_name}_contexts.txt'
        print(f"Saving {file_path}")
        with open(file_path, 'w') as fp:
            for context in contexts:
                context = ' '.join(context).replace('\n', '<P>')
                fp.write(context + '\n')

        # - save stats -
        file_path = directory / f'{file_name}_stats.json'
        print(f"Saving {file_path}")
        with open(file_path, 'w') as fp:
            json.dump(stats, fp)
    else:
        # - load contexts -
        file_path = directory / f'{file_name}_contexts.txt'
        print(f"Loading {file_path}")
        with open(file_path, 'r') as fp:
            contexts = [c.split() for c in fp.readlines()]

    return contexts


def get_corpus(contexts, args, directory, file_name):
    if args.make_corpus:
        # - make bow corpus -
        bow_corpus, dictionary = texts2corpus(contexts, stopwords=None)

        # - save dictionary -
        file_path = directory / f'{file_name}.dict'
        print(f"Saving {file_path}")
        dictionary.save(str(file_path))

        # - save dictionary frequencies as plain text -
        dict_table = pd.Series(dictionary.token2id).to_frame(name='idx')
        dict_table['freq'] = dict_table['idx'].map(dictionary.cfs.get)
        dict_table = dict_table.reset_index()
        dict_table = dict_table.set_index('idx', drop=True).rename({'index': 'token'}, axis=1)
        dict_table = dict_table.sort_index()
        file_path = directory / f'{file_name}_dict.csv'
        print(f"Saving {file_path}")
        # dictionary.save_as_text(file_path, sort_by_word=False)
        dict_table.to_csv(file_path, sep='\t')

        # - save bow corpus -
        file_path = directory / f'{file_name}_bow.mm'
        print(f"Saving {file_path}")
        MmCorpus.serialize(str(file_path), bow_corpus)

        # - log transform and entropy-normalize corpus -
        entropy_corpus = entropy_transform(bow_corpus, dictionary)

        # - save entropy-normalized corpus -
        file_path = directory / f'{file_name}_entropy.mm'
        print(f"Saving {file_path}")
        MmCorpus.serialize(str(file_path), entropy_corpus)

        # - tfidf transform corpus -
        tfidf_corpus = tfidf_transform(bow_corpus)

        # - save entropy-normalized corpus -
        file_path = directory / f'{file_name}_tfidf.mm'
        print(f"Saving {file_path}")
        MmCorpus.serialize(str(file_path), tfidf_corpus)

        if args.tfidf:
            file_name += '_tfidf'
            corpus = tfidf_corpus
            del entropy_corpus
        else:
            file_name += '_entropy'
            corpus = entropy_corpus
            del tfidf_corpus
    else:
        # - load dictionary -
        file_path = directory / f'{file_name}.dict'
        print(f"Loading dictionary from {file_path}")
        dictionary = Dictionary.load(str(file_path))

        # - load corpus -
        if args.tfidf:
            file_name += '_tfidf'
            file_path = directory / f'{file_name}.mm'
        else:
            file_name += '_entropy'
            file_path = directory / f'{file_name}.mm'
        print(f"Loading corpus from {file_path}")
        corpus = MmCorpus(str(file_path))

    return corpus, dictionary


def get_document_vectors(corpus, dictionary, args, directory, file_name):
    if args.make_lsi:
        model, document_vectors, term_vectors = lsi_transform(
            corpus=corpus, dictionary=dictionary, nb_topics=args.nb_topics,
            use_callbacks=False, cache_in_memory=True
        )

        # --- save model ---
        file_path = directory / f'{file_name}_lsi.model'
        print(f"Saving model to {file_path}")
        model.save(str(file_path))

        # --- save document vectors ---
        file_path = directory / f'{file_name}_lsi_document_vectors.csv'
        print(f"Saving document vectors to {file_path}")
        document_vectors.to_csv(file_path)

        # --- save term vectors ---
        file_path = directory / f'{file_name}_lsi_term_vectors.csv'
        print(f"Saving document vectors to {file_path}")
        term_vectors.to_csv(file_path)
    else:
        # --- load document vectors ---
        file_path = directory / f'{file_name}_lsi_document_vectors.csv'
        print(f"Loading document vectors from {file_path}")
        document_vectors = pd.read_csv(file_path, index_col=0)

    return document_vectors


def main():
    args = parse_args()
    print(args)

    file_name = f'{args.dataset}_{args.version}'
    directory = SEMD_PATH / args.version
    directory.mkdir(exist_ok=True, parents=True)

    contexts = get_contexts(args, directory, file_name)
    corpus, dictionary = get_corpus(contexts, args, directory, file_name)
    document_vectors = get_document_vectors(corpus, dictionary, args, directory, file_name)

    # --- calculate semd for vocabulary ---
    if args.terms:
        terms_path = Path(args.terms).resolve()
        with open(terms_path) as fp:
            terms = [line.strip() for line in fp.readlines()]
            print(terms)
        file_path = terms_path.with_suffix('.semd')
    else:
        terms = dictionary.token2id.keys()
        file_path = directory / f'{file_name}.semd'
    semd_values = calculate_semantic_diversity(terms, dictionary, corpus, document_vectors.values)

    # - save SemD values for vocabulary -
    print(f"Saving SemD values to {file_path}")
    semd_values.to_csv(file_path)

    print(semd_values)


if __name__ == '__main__':
    main()
