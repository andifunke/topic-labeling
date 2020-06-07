import argparse
import gc
import json
import re
from collections import Counter
from random import shuffle
from time import time
from typing import Iterable, List

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.matutils import corpus2dense, corpus2csc
from gensim.models import TfidfModel, LsiModel
from tqdm import tqdm

from topic_labeling.topic_modeling.lda import split_corpus
from topic_labeling.utils.constants import (
    SIMPLE_PATH, POS, TOKEN, HASH, PUNCT, DATASETS, GOOD_IDS, WORD_PATTERN,
    POS_N, POS_NV, POS_NVA, SEMD_PATH
)
from topic_labeling.utils.utils import init_logging, log_args


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

    parser.add_argument('--tfidf', dest='tfidf', action='store_true', required=False)
    parser.add_argument('--no-tfidf', dest='tfidf', action='store_false', required=False)
    parser.set_defaults(tfidf=False)

    parser.add_argument('--make-contexts', dest='make_contexts', action='store_true')
    parser.add_argument('--load-contexts', dest='make_contexts', action='store_false')
    parser.set_defaults(make_contexts=True)

    parser.add_argument('--make-corpus', dest='make_corpus', action='store_true')
    parser.add_argument('--load-corpus', dest='make_corpus', action='store_false')
    parser.set_defaults(make_corpus=True)

    parser.add_argument('--make-lsi', dest='make_lsi', action='store_true')
    parser.add_argument('--load-lsi', dest='make_lsi', action='store_false')
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


def calculate_semantic_diversity(dictionary, corpus, document_vectors, contexts):

    def _per_word_calculation(word):
        pass

    csc_matrix = corpus2csc(corpus, dtype=np.float32)
    csc_matrix_t = csc_matrix.transpose()

    for i, term in enumerate(dictionary.token2id.keys()):
        term_id = dictionary.token2id[term]
        assert i == term_id, f'{i}, {term}, {term_id}'
        term_docs = csc_matrix_t.getrow(term_id)
        term_docs2 = term_docs.nonzero()[1]
        print(term_docs2)
        term_docs3 = []
        for j, context in enumerate(contexts):
            if term in context:
                print(j, context)
                term_docs3.append(j)
        print(term_docs3)
        print()
        # TODO: work in progress



def lsi_transform(corpus, dictionary, nb_topics=300, use_callbacks=False, cache_in_memory=False):
    _split_ = '_split' if use_callbacks else ''

    # corpus = MmCorpus(data_file)
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


def entropy_transform(corpus, dictionary):
    # take the log
    # sparse_matrix = corpus2csc(corpus)
    # print(sparse_matrix)
    # sparse_matrix.data = np.log(sparse_matrix.data)
    # print(sparse_matrix)

    # calculate "entropy"
    # TODO: While this is exactly the algorithm described in the paper it is certainly not right.
    entropy = Counter()
    for context in corpus:
        for index, value in context:
            corpus_freq = dictionary.dfs[index]
            p_c = value / corpus_freq
            ic = p_c * np.log(p_c)
            # print(index, value, corpus_freq, p_c, ic)
            entropy[index] -= ic
            if entropy[index] == 0:
                # TODO: this case should actually be avoided by the removal of infrequent words.
                print(index, value, dictionary.id2token[index])
                print()

    # calculate transformed value
    entropy_corpus = [[(i, np.log(v) / entropy[i]) for i, v in context] for context in corpus]

    return entropy_corpus


def tfidf_transform(bow_corpus):
    tfidf_model = TfidfModel(bow_corpus)
    tfidf_corpus = tfidf_model[bow_corpus]
    return tfidf_corpus


def calculate_chunks(df, window):
    print(f"Calculating chunks")
    # TODO: this shouldn't take so long
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


def main():
    args = parse_args()

    corpus_type = 'tfidf' if args.tfidf else 'entropy'
    logger = init_logging(
        name=f"MM_{args.dataset}_{corpus_type}", basic=False, to_stdout=True, to_file=True
    )
    log_args(logger, args)

    file_name = f'{args.dataset}_{args.version}'
    directory = SEMD_PATH / args.version
    directory.mkdir(exist_ok=True, parents=True)

    # --- Make of Load the contexts ---
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

    # --- Make or Load a Matrix Market corpus which will be either entropy-normalized
    #     or tf-idf transformed ---
    if args.make_corpus:
        # - make bow corpus -
        corpus, dictionary = texts2corpus(contexts, stopwords=None)

        # - save dictionary -
        file_path = directory / f'{file_name}.dict'
        print(f"Saving {file_path}")
        dictionary.save(str(file_path))

        # - save bow corpus -
        file_path = directory / f'{file_name}_bow.mm'
        print(f"Saving {file_path}")
        MmCorpus.serialize(str(file_path), corpus)

        # - log transform and entropy-normalize corpus -
        entropy_corpus = entropy_transform(corpus, dictionary)

        # - save entropy-normalized corpus -
        file_path = directory / f'{file_name}_entropy.mm'
        print(f"Saving {file_path}")
        MmCorpus.serialize(str(file_path), entropy_corpus)

        # - tfidf transform corpus -
        tfidf_corpus = tfidf_transform(corpus)

        # - save entropy-normalized corpus -
        file_path = directory / f'{file_name}_tfidf.mm'
        print(f"Saving {file_path}")
        MmCorpus.serialize(str(file_path), entropy_corpus)

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

    # --- apply LSI ---
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
        document_vectors = pd.read_csv(file_path)

    # --- calculate semd for vocabulary ---
    semd_values = calculate_semantic_diversity(dictionary, corpus, document_vectors, contexts)

    # - save SemD values for vocabulary -
    file_path = directory / f'{file_name}_semd_values.csv'
    print(f"Saving SemD values to {file_path}")
    semd_values.to_csv(file_path)

    print(semd_values)


if __name__ == '__main__':
    main()
