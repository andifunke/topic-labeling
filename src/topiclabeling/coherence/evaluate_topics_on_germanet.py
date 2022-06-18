from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from pygermanet import load_germanet, Synset
from tqdm import tqdm

from topiclabeling.utils.constants import LDA_DIR
from topiclabeling.coherence.evaluate_topics import parse_args
from topiclabeling.utils.utils import load
from topiclabeling.utils.logging import init_logging, log_args, logg

np.set_printoptions(precision=3)
gn = load_germanet()
tqdm.pandas()


sim_fns = {"lch": Synset.sim_lch, "res": Synset.sim_res, "lin": Synset.sim_lin}
dst_fns = {"jcn": Synset.dist_jcn}


def orth(synset):
    return synset.lemmas[0].orthForm


def compare_synset_lists(synset_list1, synset_list2, sim_func, agg_func):
    try:
        return agg_func(
            sim_func(ss1, ss2) for ss1 in synset_list1 for ss2 in synset_list2
        )
    except ValueError:
        return np.nan


def similarity(topic, topn, sim_fn=None, dist_fn=None):
    arr = np.empty((topn, topn))
    arr[:] = np.nan
    topic = topic.to_numpy()

    for i, ssl1 in enumerate(topic):
        for j, ssl2 in enumerate(topic[i + 1 :], i + 1):
            arr[i, j] = compare_synset_lists(
                ssl1,
                ssl2,
                sim_func=dist_fn if dist_fn is not None else sim_fn,
                agg_func=min if dist_fn is not None else max,
            )

    if np.isnan(arr).all():
        return np.nan

    sim = np.nanmean(arr)
    sim = -sim if dist_fn is not None else sim

    return sim


def main():
    args = parse_args()

    init_logging(
        name=f'Eval_topics_on_germanet{f"_{args.dataset}" if args.dataset else ""}',
        to_stdout=True,
        to_file=True,
    )
    log_args(args)

    purpose = "rerank" if args.rerank else "topics"
    if args.topics:
        topics = pd.read_csv(args.topics, header=None)
        topics.columns = [f"term{i}" for i in topics.columns]
    elif args.topics:
        topics = load(
            purpose,
            args.dataset,
            args.version,
            args.corpus_type,
            args.lsi,
            *args.params,
            *args.nbtopics,
        )
    else:
        raise ValueError("Either --dataset or --topics is required.")

    if args.topn > 0:
        topics = topics[: args.topn]
    else:
        args.topn = topics.shape[1]
    logg(f"Number of topics {topics.shape}")

    logg("Getting SynSets for topic terms")
    topic_synsets = topics.applymap(gn.synsets)

    for metric in ["lch", "res", "jcn", "lin"]:
        logg(f"metric: {metric}", flush=True)
        topics[metric] = topic_synsets.progress_apply(
            similarity,
            axis=1,
            sim_fn=sim_fns.get(metric),
            dist_fn=dst_fns.get(metric),
            topn=args.topn,
        )

    if args.topics:
        tpx_path = Path(args.topics)
        if args.rerank:
            file = tpx_path.parent / f"{tpx_path.stem}_reranker-eval_germanet.csv"
        else:
            file = tpx_path.parent / f"{tpx_path.stem}_topic-scores_germanet.csv"
    else:
        tpx_path = LDA_DIR / args.version / "bow" / "topics"
        if args.rerank:
            file = tpx_path / f"{args.dataset}_reranker-eval_germanet.csv"
        else:
            file = (
                tpx_path / f'{args.dataset}{f"_{args.lsi}" if args.lsi else ""}_'
                f"{args.version}_{args.corpus_type}_topic-scores_germanet.csv"
            )

    if file.exists():
        file = file.parent / file.name.replace(
            ".csv", f'_{str(time()).split(".")[0]}.csv'
        )

    logg(f"Writing {file}")
    topics.to_csv(file)
    logg("done")


if __name__ == "__main__":
    main()
