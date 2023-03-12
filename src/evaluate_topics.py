import argparse
import gc
from os.path import join, exists
from time import time

import numpy as np
import pandas as pd
from gensim.models import CoherenceModel

from constants import PARAMS, NBTOPICS, DATASETS, LDA_PATH, DSETS
from utils import init_logging, load, log_args
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def cosine_similarities(vector_1, vectors_all):
    norm = np.linalg.norm(vector_1)
    all_norms = np.linalg.norm(vectors_all, axis=1)
    dot_products = np.dot(vectors_all, vector_1)
    similarities = dot_products / (norm * all_norms)
    return similarities


def pairwise_similarity(topic, kvs, ignore_oov=True):
    similarities = dict()
    for name, kv in kvs.items():
        vector = lambda x: kv[x] if x in kv else np.nan
        vectors = topic.map(vector).dropna()
        if len(vectors) < 2:
            similarities[name] = np.nan
            continue
        vectors = vectors.apply(pd.Series).values
        sims = np.asarray([cosine_similarities(vec, vectors) for vec in vectors]).mean(
            axis=0
        )
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
        vector = lambda x: kv[x] if x in kv else np.nan
        vectors = topic.map(vector).dropna()
        if len(vectors) < 2:
            similarities[name] = np.nan
            continue
        vectors = vectors.apply(pd.Series).values
        mean_vec = np.mean(vectors, axis=0)
        similarity = cosine_similarities(mean_vec, vectors).mean()
        similarities[name] = similarity
    return pd.Series(similarities)


def eval_coherence(
    topics,
    dictionary,
    corpus=None,
    texts=None,
    keyed_vectors=None,
    metrics=None,
    window_size=None,
    suffix="",
    cores=1,
    logg=print,
    topn=10,
):
    if not (corpus or texts or keyed_vectors):
        logg("provide corpus, texts and/or keyed_vectors")
        return
    if metrics is None:
        if corpus is not None:
            metrics = ["u_mass"]
        if texts is not None:
            if metrics is None:
                metrics = ["c_v", "c_npmi", "c_uci"]
            else:
                metrics += ["c_v", "c_npmi", "c_uci"]
        if keyed_vectors is not None:
            if metrics is None:
                metrics = ["c_w2v"]
            else:
                metrics += ["c_w2v"]

    # add out of vocabulariy terms dictionary and documents
    in_dict = topics.applymap(lambda x: x in dictionary.token2id)
    oov = topics[~in_dict]
    oov = oov.apply(set)
    oov = set().union(*oov)
    isstr = lambda x: isinstance(x, str)
    tolist = lambda x: [x]
    oov = sorted(map(tolist, filter(isstr, oov)))
    logg(f"OOV: {oov}")
    if oov:
        dictionary.add_documents(oov, prune_at=None)
        _ = dictionary[0]

    scores = dict()
    topics_values = topics.values
    for metric in metrics:
        t0 = time()
        gc.collect()
        logg(metric)
        txt = texts + oov if texts else None
        cm = CoherenceModel(
            topics=topics_values,
            dictionary=dictionary,
            corpus=corpus,
            texts=txt,
            coherence=metric,
            topn=topn,
            window_size=window_size,
            processes=cores,
            keyed_vectors=keyed_vectors,
        )
        coherence_scores = cm.get_coherence_per_topic(with_std=True, with_support=True)
        scores[metric + suffix] = coherence_scores
        gc.collect()
        t1 = int(time() - t0)
        logg(
            "    done in {:02d}:{:02d}:{:02d}".format(
                t1 // 3600, (t1 // 60) % 60, t1 % 60
            )
        )

    df = pd.DataFrame(scores)
    df.index = topics.index
    gc.collect()
    return df


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--version", type=str, required=False, default="noun")
    parser.add_argument("--tfidf", dest="tfidf", action="store_true", required=False)
    parser.add_argument(
        "--no-tfidf", dest="tfidf", action="store_false", required=False
    )
    parser.set_defaults(tfidf=False)
    parser.add_argument("--rerank", dest="rerank", action="store_true", required=False)
    parser.add_argument(
        "--no-rerank", dest="rerank", action="store_false", required=False
    )
    parser.set_defaults(rerank=False)
    parser.add_argument("--lsi", dest="lsi", action="store_true", required=False)
    parser.add_argument("--no-lsi", dest="lsi", action="store_false", required=False)
    parser.set_defaults(lsi=False)
    parser.add_argument("--params", nargs="*", type=str, required=False, default=PARAMS)
    parser.add_argument(
        "--nbtopics", nargs="*", type=int, required=False, default=NBTOPICS
    )
    parser.add_argument("--topn", type=int, required=False, default=-1)
    parser.add_argument("--cores", type=int, required=False, default=4)
    parser.add_argument(
        "--method",
        type=str,
        required=False,
        default="both",
        choices=["coherence", "w2v", "both"],
    )

    args = parser.parse_args()

    args.dataset = DSETS.get(args.dataset, args.dataset)
    corpus_type = "tfidf" if args.tfidf else "bow"
    lsi = "lsi" if args.lsi else ""
    use_coherence = args.method in ["coherence", "both"]
    use_w2v = args.method in ["w2v", "both"]

    return (
        args.dataset,
        args.version,
        args.params,
        args.nbtopics,
        args.topn,
        args.cores,
        corpus_type,
        use_coherence,
        use_w2v,
        args.rerank,
        lsi,
        args,
    )


def main():
    (
        dataset,
        version,
        params,
        nbtopics,
        topn,
        cores,
        corpus_type,
        use_coherence,
        use_w2v,
        rerank,
        lsi,
        args,
    ) = parse_args()

    logger = init_logging(
        name=f"Eval_topics_{dataset}", basic=False, to_stdout=True, to_file=True
    )
    log_args(logger, args)
    logg = logger.info

    purpose = "rerank" if rerank else "topics"
    topics = load(
        purpose, dataset, version, corpus_type, lsi, *params, *nbtopics, logg=logg
    )
    if topn > 0:
        topics = topics[:topn]
    else:
        topn = topics.shape[1]
    logg(f"number of topics: {topics.shape}")
    unique_topics = topics.drop_duplicates()
    logg(f"number of unique topics: {unique_topics.shape}")
    wiki_dict = load("dict", "dewiki", "unfiltered", logg=logg)

    dfs = []
    if use_coherence:
        dictionary = load("dict", dataset, version, corpus_type, logg=logg)
        corpus = load("corpus", dataset, version, corpus_type, logg=logg)
        texts = load("texts", dataset, version, logg=logg)

        df = eval_coherence(
            topics=unique_topics,
            dictionary=dictionary,
            corpus=corpus,
            texts=texts,
            keyed_vectors=None,
            metrics=None,
            window_size=None,
            suffix="",
            cores=cores,
            logg=logg,
            topn=topn,
        )
        del dictionary, corpus, texts
        gc.collect()
        dfs.append(df)

        wiki_texts = load("texts", "dewiki", logg=logg)
        df = eval_coherence(
            topics=unique_topics,
            dictionary=wiki_dict,
            corpus=None,
            texts=wiki_texts,
            keyed_vectors=None,
            metrics=None,
            window_size=None,
            suffix="_wikt",
            cores=cores,
            logg=logg,
            topn=topn,
        )
        gc.collect()
        dfs.append(df)

        df = eval_coherence(
            unique_topics,
            wiki_dict,
            corpus=None,
            texts=wiki_texts,
            keyed_vectors=None,
            metrics=["c_uci"],
            window_size=20,
            suffix="_wikt_w20",
            cores=cores,
            logg=logg,
            topn=topn,
        )
        del wiki_texts
        gc.collect()
        dfs.append(df)

    df_sims = None
    if use_w2v:
        d2v = load("d2v", logg=logg).docvecs
        w2v = load("w2v", logg=logg).wv
        ftx = load("ftx", logg=logg).wv
        # Dry run to make sure both indices are fully in RAM
        d2v.init_sims()
        _ = d2v.vectors_docs_norm[0]
        w2v.init_sims()
        _ = w2v.vectors_norm[0]
        ftx.init_sims()
        _ = ftx.vectors_norm[0]

        df = eval_coherence(
            topics=unique_topics,
            dictionary=wiki_dict,
            corpus=None,
            texts=None,
            keyed_vectors=w2v,
            metrics=None,
            window_size=None,
            suffix="_w2v",
            cores=cores,
            logg=logger.info,
            topn=topn,
        )
        gc.collect()
        dfs.append(df)

        df = eval_coherence(
            topics=unique_topics,
            dictionary=wiki_dict,
            corpus=None,
            texts=None,
            keyed_vectors=ftx,
            metrics=None,
            window_size=None,
            suffix="_ftx",
            cores=cores,
            logg=logger.info,
            topn=topn,
        )
        gc.collect()
        dfs.append(df)

        # apply custom similarity metrics
        kvs = {"d2v": d2v, "w2v": w2v, "ftx": ftx}
        ms = unique_topics.apply(lambda x: mean_similarity(x, kvs), axis=1)
        ps = unique_topics.apply(
            lambda x: pairwise_similarity(x, kvs, ignore_oov=True), axis=1
        )
        ps2 = unique_topics.apply(
            lambda x: pairwise_similarity(x, kvs, ignore_oov=False), axis=1
        )
        df_sims = pd.concat(
            {
                "mean_similarity": ms,
                "pairwise_similarity_ignore_oov": ps,
                "pairwise_similarity": ps2,
            },
            axis=1,
        )
        del d2v, w2v, ftx
        gc.collect()

    dfs = pd.concat(dfs, axis=1)
    dfs = (
        dfs.stack()
        .apply(pd.Series)
        .rename(columns={0: "score", 1: "stdev", 2: "support"})
        .unstack()
    )
    if df_sims is not None:
        dfs = pd.concat([dfs, df_sims], axis=1)

    # restore scores for all topics from results of unique topics
    topics.columns = pd.MultiIndex.from_tuples(
        [("terms", t) for t in list(topics.columns)]
    )
    topic_columns = list(topics.columns)
    fillna = lambda grp: grp.fillna(method="ffill") if len(grp) > 1 else grp
    dfs = (
        topics.join(dfs)
        .groupby(topic_columns)
        .apply(fillna)
        .drop(topic_columns, axis=1)
    )

    tpx_path = join(LDA_PATH, version, "bow", "topics")
    if rerank:
        file = join(tpx_path, f"{dataset}_reranker-eval.csv")
    else:
        file = join(
            tpx_path,
            f'{dataset}{"_"+lsi if lsi else ""}_{version}_{corpus_type}_topic-scores.csv',
        )
    if exists(file):
        file = file.replace(".csv", f'_{str(time()).split(".")[0]}.csv')

    logg(f"Writing {file}")
    dfs.to_csv(file)
    logg("done")

    return dfs


if __name__ == "__main__":
    main()
