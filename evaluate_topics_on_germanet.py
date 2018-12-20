import argparse
from os.path import join, exists
from time import time

import numpy as np
from pygermanet import load_germanet, Synset
from tqdm import tqdm

from constants import LDA_PATH, PARAMS, NBTOPICS, DSETS
from utils import load, init_logging, log_args

np.set_printoptions(precision=3)
gn = load_germanet()
tqdm.pandas()


def orth(synset):
    return synset.lemmas[0].orthForm


def compare_synset_lists(synset_list1, synset_list2, sim_func, agg_func):
    try:
        return agg_func(sim_func(ss1, ss2) for ss1 in synset_list1 for ss2 in synset_list2)
    except ValueError:
        return np.nan


def similarities(topic, ignore_unknown=True, sim_func=Synset.sim_lch, agg_func=max, topn=10):
    arr = np.zeros((topn, topn))
    for j, ssl1 in enumerate(topic.values):
        for k, ssl2 in enumerate(topic.values[j+1:], j+1):
            arr[j, k] = compare_synset_lists(ssl1, ssl2, sim_func, agg_func)
    arr = np.add(arr, arr.T)
    if ignore_unknown:
        arr[arr == 0] = np.nan
    return np.nanmean(arr)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--version", type=str, required=False, default='noun')
    parser.add_argument('--tfidf', dest='tfidf', action='store_true', required=False)
    parser.add_argument('--no-tfidf', dest='tfidf', action='store_false', required=False)
    parser.set_defaults(tfidf=False)
    parser.add_argument('--rerank', dest='rerank', action='store_true', required=False)
    parser.add_argument('--no-rerank', dest='rerank', action='store_false', required=False)
    parser.set_defaults(rerank=False)
    parser.add_argument("--params", nargs='*', type=str, required=False, default=PARAMS)
    parser.add_argument("--nbtopics", nargs='*', type=int, required=False, default=NBTOPICS)
    parser.add_argument("--topn", type=int, required=False, default=10)

    args = parser.parse_args()

    args.dataset = DSETS.get(args.dataset, args.dataset)
    corpus_type = "tfidf" if args.tfidf else "bow"

    return (
        args.dataset, args.version, corpus_type, args.params, args.nbtopics, args.topn, args.rerank, args
    )


def main():
    dataset, version, corpus_type, params, nbtopics, topn, rerank, args = parse_args()

    logger = init_logging(name=f'Eval_topics_on_germanet_{dataset}', basic=False, to_stdout=True, to_file=True)
    log_args(logger, args)
    logg = logger.info

    purpose = 'rerank' if rerank else 'topics'
    topics = load(purpose, dataset, version, corpus_type, *params, *nbtopics)
    logg(f'Number of topics {len(topics)}')

    logg('Getting SynSets for topic terms')
    sstopics = topics.applymap(gn.synsets)

    topics['lch'] = sstopics.progress_apply(
        similarities, axis=1, sim_func=Synset.sim_lch, agg_func=max, topn=topn
    )
    topics['lch_ignr_unkwn'] = sstopics.progress_apply(
        similarities, axis=1, sim_func=Synset.sim_lch, agg_func=max, topn=topn, ignore_unknown=False
    )
    topics['res'] = sstopics.progress_apply(
        similarities, axis=1, sim_func=Synset.sim_res, agg_func=max, topn=topn
    )
    topics['res_ignr_unkwn'] = sstopics.progress_apply(
        similarities, axis=1, sim_func=Synset.sim_res, agg_func=max, topn=topn, ignore_unknown=False
    )
    topics['jcn'] = sstopics.progress_apply(
        similarities, axis=1, sim_func=Synset.dist_jcn, agg_func=min, topn=topn
    )
    topics['jcn_ignr_unkwn'] = sstopics.progress_apply(
        similarities, axis=1, sim_func=Synset.dist_jcn, agg_func=min, topn=topn, ignore_unknown=False
    )
    topics['lin'] = sstopics.progress_apply(
        similarities, axis=1, sim_func=Synset.sim_lin, agg_func=max, topn=topn
    )
    topics['lin_ignr_unkwn'] = sstopics.progress_apply(
        similarities, axis=1, sim_func=Synset.sim_lin, agg_func=max, topn=topn, ignore_unknown=False
    )

    topics = topics.iloc[:, topn:]
    tpx_path = join(LDA_PATH, version, corpus_type, 'topics')
    if rerank:
        file = join(tpx_path, f'{dataset}_reranker-eval_germanet.csv')
    else:
        file = join(tpx_path, f'{dataset}_{version}_{corpus_type}_topic-scores_germanet.csv')
    if exists(file):
        file = file.replace('.csv', f'_{str(time()).split(".")[0]}.csv')

    logg(f'Writing {file}')
    topics.to_csv(file)


if __name__ == '__main__':
    main()
