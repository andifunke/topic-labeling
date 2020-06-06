import argparse
import gc
import json
import re
from collections import Counter
from random import shuffle
from typing import Iterable, List

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.matutils import corpus2dense, corpus2csc
from gensim.models import TfidfModel, LsiModel

from topic_labeling.topic_modeling.lda import split_corpus
from topic_labeling.utils.constants import (
    SIMPLE_PATH, POS, TOKEN, HASH, PUNCT, DATASETS, GOOD_IDS, WORD_PATTERN,
    POS_N, POS_NV, POS_NVA, SEMD_PATH
)
from topic_labeling.utils.utils import init_logging, log_args


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


def texts2corpus(contexts, stopwords=None, min_contexts=40, filter_above=1, keep_n=200_000):
    print(f"Generating bow corpus and dictionary")

    dictionary = Dictionary(contexts, prune_at=None)
    dictionary.filter_extremes(no_below=min_contexts, no_above=filter_above, keep_n=keep_n)

    # filter some noise (e.g. special characters)
    if stopwords:
        stopword_ids = [dictionary.token2id[token] for token in stopwords]
        dictionary.filter_tokens(bad_ids=stopword_ids, good_ids=None)

    bow_corpus = [dictionary.doc2bow(text) for text in contexts]

    return bow_corpus, dictionary


def chunks_from_documents(documents: Iterable, window_size: int) -> List:
    contexts = []
    for document in documents:
        if len(document) > window_size:
            chunks = [document[x:x+window_size] for x in range(0, len(document), window_size)]
            contexts += chunks
        else:
            contexts.append(document)

    return contexts


def make_contexts(dataset, nb_files, pos_tags, window_size=1000):
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
        print(f"Processing {nb_files} files")

    nb_words = 0
    texts = []
    for filename in files:
        gc.collect()

        print(f"Reading {filename}")
        df = pd.read_pickle(filename)
        print(f"    initial number of words: {len(df)}")

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

        # df = df[df.token.str.len() > 1]
        # df = df[~df.token.isin(BAD_TOKENS)]

        # TODO: do we want to remove non-work tokens?
        df = df[df.token.str.match(WORD_PATTERN)]
        nb_words += len(df)
        print(f"    remaining number of words: {len(df)}")

        # groupby sorts the documents by hash-id
        # which is equal to shuffling the dataset before building the model
        df = df.groupby([HASH], sort=True)[TOKEN].agg(docs_to_lists)
        documents = df.values.tolist()
        print(f"    number of documents: {len(documents)}")
        if window_size > 0:
            contexts = chunks_from_documents(documents, window_size)
        else:
            contexts = documents

        print(f"    number of contexts: {len(contexts)}")
        texts += contexts

    # re-shuffle documents
    if len(files) > 1:
        shuffle(texts)
    if nb_files is not None:
        nb_files = min(nb_files, len(files))

    nb_docs = len(texts)
    print(f"Total number of documents: {nb_docs}")
    print(f"Total number of words: {nb_words}")
    stats = dict(dataset=dataset, pos_set=sorted(pos_tags), nb_docs=nb_docs, nb_words=nb_words)
    return texts, stats, nb_files


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


def main():
    args = parse_args()

    corpus_type = 'tfidf' if args.tfidf else 'entropy'
    logger = init_logging(
        name=f"MM_{args.dataset}_{corpus_type}", basic=False, to_stdout=True, to_file=True
    )
    logg = logger.info if logger else print
    log_args(logger, args)

    file_name = f'{args.dataset}_{args.version}'
    directory = SEMD_PATH / args.version
    directory.mkdir(exist_ok=True, parents=True)

    # --- Make of Load the contexts ---
    if args.make_contexts:
        # - make contexts -
        contexts, stats, nb_files = make_contexts(
            args.dataset, args.nb_files, args.pos_tags, window_size=args.window
        )

        if args.min_word_freq > 0:
            # TODO: The approach for removal of infrequent words needs to be reconsidered.
            #       It should ideally be applied after the dataset is initially read.
            contexts = remove_infrequent_words(contexts, args.min_word_freq)

        gc.collect()

        # - save contexts -
        file_path = directory / f'{file_name}_texts.json'
        logg(f"Saving {file_path}")
        with open(file_path, 'w') as fp:
            json.dump(contexts, fp, ensure_ascii=False)

        # - save stats -
        file_path = directory / f'{file_name}_stats.json'
        print(f"Saving {file_path}")
        with open(file_path, 'w') as fp:
            json.dump(stats, fp)
    else:
        # - load contexts -
        file_path = directory / f'{file_name}_contexts.json'
        print(f"Loading {file_path}")
        with open(file_path, 'r') as fp:
            contexts = json.load(fp)

    # --- Make or Load a Matrix Market corpus which will be either entropy-normalized
    #     or tf-idf transformed ---
    if args.make_corpus:
        # - make bow corpus -
        corpus, dictionary = texts2corpus(
            contexts, stopwords=None, min_contexts=args.min_contexts, filter_above=1
        )

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
