import argparse
import gc
import warnings
from logging import WARNING
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import CoherenceModel, Doc2Vec
from tqdm import tqdm

from topiclabeling.utils.constants import PARAMS, LDA_DIR, DATASETS_FULL, NB_TOPICS
from topiclabeling.utils.logging import init_logging, log_args, logg
from topiclabeling.utils.utils import load

# warnings.simplefilter(action='ignore', category=FutureWarning)


def cosine_similarities(vector_1, vectors_all):
    norm = np.linalg.norm(vector_1)
    all_norms = np.linalg.norm(vectors_all, axis=1)
    dot_products = np.dot(vectors_all, vector_1)
    similarities = dot_products / (norm * all_norms)
    return similarities


def pairwise_similarity(topic, kvs, ignore_oov=True):
    similarities = dict()
    for name, kv in kvs.items():
        vectors = topic.map(lambda x: kv[x] if x in kv else np.nan).dropna()
        if len(vectors) < 2:
            similarities[name] = np.nan
            continue
        vectors = vectors.apply(pd.Series).values
        sims = np.asarray([cosine_similarities(vec, vectors) for vec in vectors]).mean(axis=0)
        if not ignore_oov:
            missing = len(topic) - len(sims)
            if missing > 0:
                sims = np.append(sims, np.zeros(missing))
        similarity = sims.mean()
        similarities[name] = similarity
    return pd.Series(similarities)


def mean_similarity(topic, kvs):
    similarities = dict()
    for name, kv in kvs.items():
        vectors = topic.map(lambda x: kv[x] if x in kv else np.nan).dropna()
        if len(vectors) < 2:
            similarities[name] = np.nan
            continue
        vectors = vectors.apply(pd.Series).values
        mean_vec = np.mean(vectors, axis=0)
        similarity = cosine_similarities(mean_vec, vectors).mean()
        similarities[name] = similarity
    return pd.Series(similarities)


def eval_coherence(
        topics, dictionary, corpus=None, texts=None, keyed_vectors=None, metrics=None,
        window_size=None, suffix='', cores=1, topn=10
):
    if not (corpus or texts or keyed_vectors):
        logg('provide corpus, texts and/or keyed_vectors')
        return

    if metrics is None:
        if corpus is not None:
            metrics = [
                'u_mass'
            ]
        if texts is not None:
            if metrics is None:
                metrics = [
                    'c_v',
                    'c_npmi',
                    'c_uci',
                ]
            else:
                metrics += [
                    'c_v',
                    'c_npmi',
                    'c_uci'
                ]
        if keyed_vectors is not None:
            if metrics is None:
                metrics = ['c_w2v']
            else:
                metrics += ['c_w2v']

    # add out of vocabulary terms dictionary and documents
    in_dict = topics.applymap(lambda x: x in dictionary.token2id)
    oov = topics[~in_dict]
    oov = oov.apply(set)
    oov = set().union(*oov)
    is_str = lambda x: isinstance(x, str)
    to_list = lambda x: [x]
    oov = sorted(map(to_list, filter(is_str, oov)))
    logg(f'OOV: {oov}')
    if oov:
        dictionary.add_documents(oov, prune_at=None)
        _ = dictionary[0]

    scores = dict()
    topics_values = topics.values
    for metric in metrics:
        t0 = time()
        gc.collect()
        logg(metric)

        if isinstance(corpus, Path):
            logg(f'Loading corpus from {corpus}')
            mm_corpus = MmCorpus(str(corpus))
        else:
            mm_corpus = corpus

        fp = None
        if metric == 'u_mass':
            txt = None
        elif isinstance(texts, Path):
            fp = open(texts, 'r')
            logg(f'Loading texts from {texts}')
            txt = tqdm(map(lambda x: x.strip().split(), fp), unit=' documents')
        else:
            txt = texts + oov if texts is not None else None

        cm = CoherenceModel(
            topics=topics_values,
            dictionary=dictionary,
            corpus=mm_corpus,
            texts=txt,
            coherence=metric,
            topn=topn,
            window_size=window_size,
            processes=cores,
            keyed_vectors=keyed_vectors
        )
        coherence_scores = cm.get_coherence_per_topic(with_std=True, with_support=True)
        scores[metric + suffix] = coherence_scores
        gc.collect()
        t1 = int(time() - t0)
        logg("    done in {:02d}:{:02d}:{:02d}".format(t1 // 3600, (t1 // 60) % 60, t1 % 60))

        if fp is not None:
            fp.close()

    if not scores:
        return None

    df = pd.DataFrame(scores)
    df.index = topics.index
    gc.collect()

    return df


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--topics", type=str, required=False,
                        help="Path to a file containing groups of terms per line. "
                             "Either --dataset or --topics is required.")
    parser.add_argument("--version", type=str, required=False, default='noun')
    parser.add_argument('--tfidf', dest='tfidf', action='store_true', required=False)
    parser.add_argument('--no-tfidf', dest='tfidf', action='store_false', required=False)
    parser.set_defaults(tfidf=False)
    parser.add_argument('--rerank', dest='rerank', action='store_true', required=False)
    parser.add_argument('--no-rerank', dest='rerank', action='store_false', required=False)
    parser.set_defaults(rerank=False)
    parser.add_argument('--lsi', dest='lsi', action='store_true', required=False)
    parser.add_argument('--no-lsi', dest='lsi', action='store_false', required=False)
    parser.set_defaults(lsi=False)
    parser.add_argument("--params", nargs='*', type=str, required=False, default=PARAMS)
    parser.add_argument("--nbtopics", nargs='*', type=int, required=False, default=NB_TOPICS)
    parser.add_argument("--topn", type=int, required=False, default=-1)
    parser.add_argument("--cores", type=int, required=False, default=4)
    parser.add_argument("--method", type=str, required=False, default='both',
                        choices=['coherence', 'w2v', 'both'])

    parser.add_argument("--dictionary", type=str, required=False)
    parser.add_argument("--corpus", type=str, required=False)
    parser.add_argument("--texts", type=str, required=False)
    parser.add_argument("--d2v", type=str, required=False)

    args = parser.parse_args()

    args.dataset = DATASETS_FULL.get(args.dataset, args.dataset)
    args.corpus_type = "tfidf" if args.tfidf else "bow"
    args.lsi = "lsi" if args.lsi else ""
    args.use_coherence = (args.method in ['coherence', 'both'])
    args.use_w2v = (args.method in ['w2v', 'both'])

    return args


def main():
    args = parse_args()

    init_logging(
        name=f'Eval_topics{f"_{args.dataset}" if args.dataset else ""}',
        to_stdout=True, to_file=True
    )
    log_args(args)

    purpose = 'rerank' if args.rerank else 'topics'
    if args.topics:
        topics = pd.read_csv(args.topics, header=None)
        topics.columns = [f'term{i}' for i in topics.columns]
    else:
        topics = load(
            purpose, args.dataset, args.version, args.corpus_type, args.lsi,
            *args.params, *args.nbtopics
        )
    if args.topn > 0:
        topics = topics[:args.topn]
    else:
        args.topn = topics.shape[1]
    logg(f'Number of topics: {topics.shape}')
    unique_topics = topics.drop_duplicates()
    logg(f'Number of unique topics: {unique_topics.shape}')
    wiki_dict = load('dict', 'dewiki', 'unfiltered')

    if args.dictionary:
        logg(f'Loading dictionary from {args.dictionary}')
        dictionary = Dictionary.load(args.dictionary)
    else:
        dictionary = load('dict', args.dataset, args.version, args.corpus_type)

    dfs = []
    if args.use_coherence:

        if args.corpus:
            corpus = Path(args.corpus)
        else:
            corpus = load('corpus', args.dataset, args.version, args.corpus_type)

        if args.texts:
            texts = Path(args.texts)
        else:
            texts = load('texts', args.dataset, args.version)

        df = eval_coherence(
            topics=unique_topics, dictionary=dictionary, corpus=corpus, texts=texts,
            keyed_vectors=None, metrics=None, window_size=None,
            suffix='', cores=args.cores, topn=args.topn,
        )
        del corpus, texts
        gc.collect()
        if df is not None:
            dfs.append(df)

        # evaluate coherence on wikipedia
        if False:
            wiki_texts = load('texts', 'dewiki')
            df = eval_coherence(
                topics=unique_topics, dictionary=wiki_dict, corpus=None, texts=wiki_texts,
                keyed_vectors=None, metrics=None, window_size=None,
                suffix='_wikt', cores=args.cores, topn=args.topn,
            )
            gc.collect()
            dfs.append(df)

            df = eval_coherence(
                unique_topics, wiki_dict, corpus=None, texts=wiki_texts,
                keyed_vectors=None, metrics=['c_uci'], window_size=20,
                suffix='_wikt_w20', cores=args.cores, topn=args.topn,
            )
            del wiki_texts
            gc.collect()
            dfs.append(df)

    df_sims = None
    if args.use_w2v:
        d2v = load('d2v').docvecs
        w2v = load('w2v').wv
        ftx = load('ftx').wv
        # Dry run to make sure both indices are fully in RAM
        d2v.init_sims()
        _ = d2v.vectors_docs_norm[0]
        w2v.init_sims()
        _ = w2v.vectors_norm[0]
        ftx.init_sims()
        _ = ftx.vectors_norm[0]

        df = eval_coherence(
            topics=unique_topics, dictionary=wiki_dict, corpus=None, texts=None,
            keyed_vectors=w2v, metrics=None, window_size=None,
            suffix='_w2v', cores=args.cores, topn=args.topn,
        )
        gc.collect()
        dfs.append(df)

        df = eval_coherence(
            topics=unique_topics, dictionary=wiki_dict, corpus=None, texts=None,
            keyed_vectors=ftx, metrics=None, window_size=None,
            suffix='_ftx', cores=args.cores, topn=args.topn,
        )
        gc.collect()
        dfs.append(df)

        # apply custom similarity metrics
        kvs = {'d2v': d2v, 'w2v': w2v, 'ftx': ftx}
        ms = unique_topics.apply(mean_similarity, kvs=kvs, axis=1)
        ps = unique_topics.apply(pairwise_similarity, kvs=kvs, ignore_oov=True, axis=1)
        ps2 = unique_topics.apply(pairwise_similarity, kvs=kvs, ignore_oov=False, axis=1)
        df_sims = pd.concat(
            {
                'mean_similarity': ms,
                'pairwise_similarity_ignore_oov': ps,
                'pairwise_similarity': ps2
            },
            axis=1
        )
        del d2v, w2v, ftx
        gc.collect()

    if args.d2v:
        d2v = Doc2Vec.load(args.d2v)
        d2v.wv.init_sims()
        _ = d2v.wv.vectors_norm[0]
        df = eval_coherence(
            topics=unique_topics, dictionary=dictionary, corpus=None, texts=None,
            keyed_vectors=d2v.wv, metrics=None, window_size=None,
            suffix='_d2v.wv.custom', cores=args.cores, topn=args.topn,
        )
        gc.collect()
        if df is not None:
            dfs.append(df)

        # apply custom similarity metrics
        kvs = {'d2v.wv.custom': d2v.wv}
        ms = unique_topics.apply(mean_similarity, kvs=kvs, axis=1)
        ps = unique_topics.apply(pairwise_similarity, kvs=kvs, ignore_oov=True, axis=1)
        ps2 = unique_topics.apply(pairwise_similarity, kvs=kvs, ignore_oov=False, axis=1)
        df_sims = pd.concat(
            {
                'mean_similarity': ms,
                'pairwise_similarity_ignore_oov': ps,
                'pairwise_similarity': ps2
            },
            axis=1
        )
        del d2v
        gc.collect()

    dfs = pd.concat(dfs, axis=1)
    dfs = (
        dfs
        .stack()
        .apply(pd.Series)
        .rename(columns={0: 'score', 1: 'stdev', 2: 'support'})
        .unstack()
    )
    if df_sims is not None:
        dfs = pd.concat([dfs, df_sims], axis=1)

    # restore scores for all topics from results of unique topics
    topics.columns = pd.MultiIndex.from_tuples([('terms', t) for t in list(topics.columns)])
    topic_columns = list(topics.columns)
    dfs = (
        topics
        .join(dfs)
        .groupby(topic_columns)
        .apply(lambda grp: grp.fillna(method='ffill') if len(grp) > 1 else grp)
        .drop(topic_columns, axis=1)
    )

    if args.topics:
        tpx_path = Path(args.topics)
        if args.rerank:
            file = tpx_path.parent / f'{tpx_path.stem}_reranker-eval.csv'
        else:
            file = tpx_path.parent / f'{tpx_path.stem}_topic-scores.csv'
    else:
        tpx_path = LDA_DIR / args.version / 'bow' / 'topics'
        if args.rerank:
            file = tpx_path / f'{args.dataset}_reranker-eval.csv'
        else:
            file = (
                tpx_path /
                f'{args.dataset}{f"_{args.lsi}" if args.lsi else ""}_'
                f'{args.version}_{args.corpus_type}_topic-scores.csv'
            )

    if file.exists():
        file = file.parent / file.name.replace('.csv', f'_{str(time()).split(".")[0]}.csv')

    logg(f'Writing {file}')
    dfs.to_csv(file)
    logg('done')

    return dfs


if __name__ == '__main__':
    main()
