# coding: utf-8
import argparse
import json
from collections import defaultdict
from os import makedirs
from os.path import join, exists
from pprint import pformat
from time import time

import numpy as np
import pandas as pd
from gensim.models import CoherenceModel
from pandas.core.common import SettingWithCopyWarning

from constants import DATASETS, METRICS, PARAMS, NBTOPICS, LDA_PATH, PLACEHOLDER
import warnings

from utils import TopicsLoader, tprint, load

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.display.precision = 3
pd.options.display.max_columns = 15
pd.options.display.width = 2000
np.set_printoptions(precision=3, threshold=None, edgeitems=None, linewidth=800, suppress=None)


# --------------------------------------------------------------------------------------------------
# --- Reranker Class ---

class Reranker(object):

    def __init__(self, topics: TopicsLoader, nb_candidate_terms=None, nb_top_terms=10, processes=-1):
        """
        :param topics:              pass instance of ModelLoader.
        :param nb_candidate_terms:  number of topic terms to evaluate the model over.
                                    nb_candidate_terms must be > nb_top_terms.
                                    The value is usually infered from the given topics.
        :param nb_top_terms:        number of remaining topic terms. The size of the final topic
                                    representation set. nb_top_terms ust be < nb_candidate_terms.
        :param processes:           number of processes used for the calculations.
        """
        self.dict_from_corpus = topics.dictionary
        self.placeholder_id = topics.dictionary.token2id[PLACEHOLDER]
        self.corpus = topics.corpus
        self.texts = topics.texts
        self.nb_topics = topics.nb_topics
        self.topic_terms = topics.topics.copy()
        self.topic_ids = topics.topic_ids()
        self.nb_top_terms = nb_top_terms
        self.processes = processes
        self.dataset = topics.dataset
        self.version = topics.version

        if nb_candidate_terms is None:
            self.nb_candidate_terms = topics.topn
        else:
            self.nb_candidate_terms = nb_candidate_terms

        # this method is only needed for the fast rerank algorithm
        # once other algorithms are implemented it should be removed from the constructor
        self.shifted_topics = self._shift_topics()
        self.topic_candidates = None
        self.eval_scores = None
        self.scores_vec = None
        self.kvs = None

        # generate some statistics
        self._statistics_ = dict()
        self._statistics_['dataset'] = topics.dataset
        self._statistics_['version'] = topics.version

    def _shift_topics(self):
        """
        from the top n terms construct all topic set that omit one term,
        resulting in n topics with n-1 topic terms for each topic
        """
        shifted_frames = []
        for i in range(self.nb_candidate_terms):
            df = pd.DataFrame(np.roll(self.topic_ids.values, shift=-i, axis=1))
            shifted_frames.append(df)
        shifted_ids = pd.concat(shifted_frames)
        # omit the first topic term, then the second and append the first etc...
        shifted_topics = shifted_ids.iloc[:, 1:].values.tolist()
        return shifted_topics

    def _init_vectors(self):
        d2v = load('d2v').docvecs
        w2v = load('w2v').wv
        ftx = load('ftx').wv

        # Dry run to make sure both indices are fully in RAM
        d2v.init_sims()
        vector = d2v.vectors_docs_norm[0]
        _ = d2v.index2entity[0]
        d2v.most_similar([vector], topn=5)

        w2v.init_sims()
        vector = w2v.vectors_norm[0]
        _ = w2v.index2entity[0]
        w2v.most_similar([vector], topn=5)

        ftx.init_sims()
        vector = ftx.vectors_norm[0]
        _ = ftx.index2entity[0]
        ftx.most_similar([vector], topn=5)

        self.kvs = {'d2v': d2v, 'w2v': w2v, 'ftx': ftx}

    def _id2term(self, id_):
        return self.dict_from_corpus[id_]

    def vote(self, df, reference, name='vote'):
        return (
            df
            .loc[:, 'term0':f'term{self.nb_top_terms - 1}']
            .apply(pd.value_counts)
            .sum(axis=1)
            [reference]
            .dropna()
            .astype(np.int16)
            .reset_index()
            .rename(columns={'index': 'term', 0: 'count'})
            .sort_values('count', ascending=False, kind='mergesort')
            [:self.nb_top_terms]
            .set_index('term')
            .squeeze()
            [reference]
            .dropna()
            .reset_index()
            .rename(lambda x: f'term{x}')
            .drop('count', axis=1)
            .squeeze()
            .rename(name)
        )

    @staticmethod
    def oop_score(col, reference_ranks):
        terms = col.values
        ref_ranks = reference_ranks[terms]
        oop = (ref_ranks - np.arange(len(ref_ranks))).abs().sum()
        return oop

    def get_reference(self):
        metric = 'ref'
        ref_topics_terms = (
            self.topic_ids.iloc[:, :self.nb_top_terms]
            .copy()
            .assign(metric=metric)
            .set_index('metric', append=True)
        )
        self._statistics_[metric] = dict()
        self._statistics_[metric]['runtime'] = 0
        return ref_topics_terms

    def _rerank_coherence_per_metric(self, metric, coherence_model=None):
        """
        Object method to trigger the reranking for a given metric.
        It uses the fast heuristic for the reranking in O(n) with n being the number
        of candidate terms. A coherence metric is applied on each set of topic terms,
        when we leave exactly one term out. The resulting coherence score indicates, if
        a term strengthens or weakens the coherence of a topic. We remove those terms
        from the set whose absence resulted in higher scores.

        :param metric:
        :param coherence_model:
        :return:
        """
        if self.shifted_topics is None:
            self.shifted_topics = self._shift_topics()

        t0 = time()
        print(f'Calculating topic candidates using {metric} coherence measure '
              f'on {self.nb_candidate_terms} candidate terms '
              f'for {self.nb_topics} topics')

        # calculate the scores for all shifted topics
        kwargs = dict(
            topics=self.shifted_topics,
            dictionary=self.dict_from_corpus,
            coherence=metric,
            topn=self.nb_candidate_terms - 1,
            processes=self.processes
        )
        if metric == 'u_mass':
            kwargs['corpus'] = self.corpus
        else:
            kwargs['texts'] = self.texts

        if coherence_model is None:
            cm = CoherenceModel(**kwargs)
        else:
            cm = coherence_model
            cm.coherence = metric

        scores1d = cm.get_coherence_per_topic()
        scores2d = np.reshape(scores1d, (self.nb_candidate_terms, -1)).T
        # the highest values indicate the terms whose absence improves the topic coherence most
        sorted_scores = np.argsort(scores2d, axis=1)
        # thus we will keep the first nbtopterms (default 10) indices
        top_scores = sorted_scores[:, :self.nb_top_terms]
        # and sort them back for convenience
        top_scores = np.sort(top_scores, axis=1)
        # replacing indices with token-ids
        tpx_ids = [self.topic_ids.values[i, top_scores[i]] for i in range(self.nb_topics)]
        tpx_ids = (
            pd.DataFrame
            .from_records(
                tpx_ids,
                columns=self.topic_terms.columns[:self.nb_top_terms],
                index=self.topic_ids.index
            )
            .assign(metric=metric)
            .set_index('metric', append=True)
        )

        t1 = int(time() - t0)
        self._statistics_[metric] = dict()
        self._statistics_[metric]['runtime'] = t1
        print("    done in {:02d}:{:02d}:{:02d}".format(t1 // 3600, (t1 // 60) % 60, t1 % 60))

        return tpx_ids

    def rerank_coherence(self, metrics=None):
        """
        Main method of a Reranker instance. It generates topic candidates for the given coherence
        metrics. A topic candidate is a reranking of the representational terms of a topic. For each
        topic each metric generates one topic candidate. This results in |topics| * (|metrics|+1)
        topic candidates, or in other words |metrics|+1 candidates for each topic. The +1 offest is
        due to the original topic ranking added to the candidate set.

        The reranking is based on the top m topic terms and then reduced to the top n topic terms
        where m > n. Typical values are m=20 and n=10. The original order of the terms is kept while
        filtering out the terms outside the n best scores.

        :param   metrics -> list of str.
                 str must be in {'u_mass', 'c_v', 'c_uci', 'c_npmi', 'vote'}.
        :return  DataFrame containing all topic candidates
        """
        available_metrics = METRICS
        if metrics is None:
            metrics = available_metrics

        print(f'Creating reranked top candidates for metrics {metrics}, using fast method')
        candidates = []

        # adding original (reference) topics
        ref_topics_terms = self.get_reference()
        candidates.append(ref_topics_terms)

        # adding several rerankings according to different metrics
        if 'u_mass' in metrics:
            umass_topics_terms = self._rerank_coherence_per_metric('u_mass')
            candidates.append(umass_topics_terms)
        if 'c_v' in metrics:
            cv_topics_terms = self._rerank_coherence_per_metric('c_v')
            candidates.append(cv_topics_terms)
        if 'c_uci' in metrics:
            cuci_topics_terms = self._rerank_coherence_per_metric('c_uci')
            candidates.append(cuci_topics_terms)
        if 'c_npmi' in metrics:
            cnpmi_topics_terms = self._rerank_coherence_per_metric('c_npmi')
            candidates.append(cnpmi_topics_terms)
        topic_candidates = pd.concat(candidates, axis=0)

        # adding candidates by majority votes from prior reference and rerankings
        if 'vote' in metrics:
            vote_topic_terms2 = (
                topic_candidates
                .groupby(level=[0, 1, 2, 3], sort=False)
                .apply(lambda x: self.vote(x, self.topic_ids.loc[x.name, :].values, name=x.name))
                .assign(metric='vote_coh')
                .set_index('metric', append=True)
            )
            topic_candidates = topic_candidates.append(vote_topic_terms2)

        topic_candidates = topic_candidates.sort_index()

        # replacing token-ids with tokens -> resulting in the final topic candidates
        topic_candidates.loc[:, 'term0':f'term{self.nb_top_terms - 1}'] = \
            topic_candidates.loc[:, 'term0':f'term{self.nb_top_terms - 1}'].applymap(self._id2term)

        def prepare_oop_score(col):
            ref_terms = self.topic_terms.loc[col.name[:-1], :]
            ref_rank = pd.Series(np.arange(self.nb_candidate_terms), index=ref_terms)
            return self.oop_score(col, ref_rank)

        topic_candidates['oop_score'] = topic_candidates.apply(prepare_oop_score, axis=1)

        if self.topic_candidates is None:
            self.topic_candidates = topic_candidates
        else:
            self.topic_candidates = self.topic_candidates.append(topic_candidates).sort_index()
        return topic_candidates

    def _rerank_vec_values(self, topic_param):

        def _rank(_df, _name):
            _df[f'{_name}_drank'] = _df[f'{_name}_dscore'].rank().map(lambda x: x - 1)
            _df[f'{_name}_rrank'] = _df[f'{_name}_rscore'].rank().map(lambda x: x - 1)
            return _df

        def _fillna_max(_df):
            _mask = _df.isnull().any(axis=1)
            _df[_mask] = _df[_mask].apply(lambda x: x.fillna(x.max()), axis=1)
            return _df

        reference = pd.Series(np.arange(self.nb_candidate_terms), index=topic_param, name='ref_rank')
        scores = [reference]
        for name, kv in self.kvs.items():
            in_kv = np.vectorize(lambda x: x in kv)
            mask = in_kv(topic_param)
            topic = topic_param[mask]
            nb_terms_in_vocab = len(topic)
            rank_scores = defaultdict(int)
            dist_scores = defaultdict(float)
            for i in range(nb_terms_in_vocab):
                entity = topic[i]
                others = np.delete(topic, i)
                distances = kv.distances(entity, tuple(others))
                argsort = distances.argsort()
                nearest = others[argsort]
                for j, term in zip(distances, others):
                    dist_scores[term] += j
                for j, term in enumerate(nearest):
                    rank_scores[term] += j
            d_score = pd.Series(dist_scores, name=f'{name}_dscore')
            r_score = pd.Series(rank_scores, name=f'{name}_rscore')
            dr = pd.concat([d_score, r_score], axis=1)
            dr = _rank(dr, name)
            scores.append(dr)
        df = pd.concat(scores, axis=1, sort=False)
        if df.isnull().any().any():
            for s in ['dscore', 'rscore', 'drank', 'rrank']:
                scols = df.columns.str.contains(s)
                df.loc[:, scols] = _fillna_max(df.loc[:, scols])

        # getting scores and ranks for all combinations -> calculating c = a+b for both distance and
        # rank scores and getting a rank for the sum
        for c, a, b in [
            ('dw', 'd2v', 'w2v'), ('df', 'd2v', 'ftx'), ('wf', 'w2v', 'ftx'), ('dwf', 'dw', 'ftx')
        ]:
            df[f'{c}_dscore'] = df[f'{a}_dscore'] + df[f'{b}_dscore']
            df[f'{c}_rscore'] = df[f'{a}_rscore'] + df[f'{b}_rscore']
            df = _rank(df, c)

        return df

    def _sort_terms(self, col):
        top_terms = col.sort_values().index.values[:self.nb_top_terms]
        col = col[col.index.isin(top_terms)]
        return col.index.values

    def _remove_not_matching_terms(self, kv_name, topic):
        kv = self.kvs[kv_name]
        # reduced_tpx = topic

        in_kv = np.vectorize(lambda x: x in kv)
        mask = in_kv(topic)
        reduced_tpx = topic[mask]
        nb_terms_in_kv = len(reduced_tpx)
        if nb_terms_in_kv > self.nb_top_terms:
            for i in range(nb_terms_in_kv - self.nb_top_terms):
                remove = kv.doesnt_match(reduced_tpx)
                reduced_tpx = reduced_tpx[reduced_tpx != remove]
        elif nb_terms_in_kv == 0:
            reduced_tpx = topic[:self.nb_top_terms]
        elif nb_terms_in_kv < self.nb_top_terms:
            nb_missing = self.nb_top_terms - nb_terms_in_kv
            for i, m in enumerate(mask):
                if not m:
                    mask[i] = True
                    nb_missing -= 1
                    if nb_missing == 0:
                        break
            reduced_tpx = topic[mask]

        ser = pd.Series(reduced_tpx, name=kv_name + '_matches')
        return ser

    def _rerank_vec_by_group(self, topic):
        topic = topic.values[0]

        df = self._rerank_vec_values(topic)
        rank_columns = [col for col in df.columns if 'rank' in col]
        df_ranks = df[rank_columns]
        reranks = df_ranks.apply(self._sort_terms, axis=0).reset_index(drop=True).T.rename(
            columns=lambda x: f'term{x}'
        )

        dred = self._remove_not_matching_terms('d2v', topic)
        wred = self._remove_not_matching_terms('w2v', topic)
        fred = self._remove_not_matching_terms('ftx', topic)
        reds = pd.concat([dred, wred, fred], axis=1).T.rename(columns=lambda x: f'term{x}')
        reranks = pd.concat([reranks, reds])

        votes = []
        for name in ['rrank', 'drank', 'matches', '']:
            subset = reranks[reranks.index.str.contains(name)]
            v = self.vote(subset, topic, f'{name}_vote_vec'.strip('_'))
            votes.append(v)
        reranks = reranks.append(votes)

        reranks['oop_score'] = reranks.apply(lambda x: self.oop_score(x, df_ranks.ref_rank), axis=1)
        return reranks

    def rerank_vec(self, topics=None):
        if topics is None:
            topics = self.topic_terms
        if self.kvs is None:
            self._init_vectors()
        topic_candidates = topics.groupby(level=[0, 1, 2, 3], sort=False).apply(self._rerank_vec_by_group)
        topic_candidates.index = topic_candidates.index.rename(names='metric', level=-1)
        self.scores_vec = topic_candidates.oop_score.unstack('metric')
        if self.topic_candidates is None:
            self.topic_candidates = topic_candidates
        else:
            self.topic_candidates = self.topic_candidates.append(topic_candidates).sort_index()
        return topic_candidates

    def rerank_greedy(self):
        pass

    def rerank_full(self):
        pass

    def reranking_statistics(self):
        self._statistics_['nb_topics'] = self.nb_topics
        self._statistics_['nb_candidate_terms'] = self.nb_candidate_terms
        self._statistics_['nb_top_terms'] = self.nb_top_terms
        self._statistics_['size_vocabulary'] = len(self.dict_from_corpus)
        self._statistics_['size_corpus'] = len(self.corpus)
        return self._statistics_

    def evaluate(self, topic_candidates=None, nbtopterms=None):
        print('evaluating topic candidates')

        # reference scores per topic for top topic terms
        if nbtopterms is None:
            nbtopterms = self.nb_top_terms

        if topic_candidates is None:
            topic_candidates = self.topic_candidates

        topic_candidates = topic_candidates.loc[:, 'term0':f'term{nbtopterms - 1}']
        topics_list = topic_candidates.values.tolist()

        print('> u_mass')
        t0 = time()
        cm_umass = CoherenceModel(
            topics=topics_list, corpus=self.corpus, dictionary=self.dict_from_corpus,
            coherence='u_mass', topn=nbtopterms, processes=self.processes
        )
        umass_scores = cm_umass.get_coherence_per_topic(with_std=False, with_support=False)
        t1 = int(time() - t0)
        print("    done in {:02d}:{:02d}:{:02d}".format(t1 // 3600, (t1 // 60) % 60, t1 % 60))

        print('> c_v')
        t0 = time()
        cm_cv = CoherenceModel(
            topics=topics_list, texts=self.texts, dictionary=self.dict_from_corpus,
            coherence='c_v', topn=nbtopterms, processes=self.processes
        )
        cv_scores = cm_cv.get_coherence_per_topic()
        t1 = int(time() - t0)
        print("    done in {:02d}:{:02d}:{:02d}".format(t1 // 3600, (t1 // 60) % 60, t1 % 60))

        # changed segmentation for c_uci and c_npmi from s_one_set to s_one_one (default)
        print('> c_uci')
        t0 = time()
        cm_cuci = CoherenceModel(
            topics=topics_list, texts=self.texts, dictionary=self.dict_from_corpus,
            coherence='c_uci', topn=nbtopterms, processes=self.processes
        )
        cuci_scores = cm_cuci.get_coherence_per_topic()
        t1 = int(time() - t0)
        print("    done in {:02d}:{:02d}:{:02d}".format(t1 // 3600, (t1 // 60) % 60, t1 % 60))

        print('> c_npmi')
        t0 = time()
        cm_cuci.coherence = 'c_npmi'  # reusing precalculated probability estimates
        cnpmi_scores1 = cm_cuci.get_coherence_per_topic()
        t1 = int(time() - t0)
        print("    done in {:02d}:{:02d}:{:02d}".format(t1 // 3600, (t1 // 60) % 60, t1 % 60))

        scores = {
            'u_mass_eval': umass_scores,
            'c_v_eval': cv_scores,
            'c_uci_eval': cuci_scores,
            'c_npmi_eval': cnpmi_scores1,
        }
        scores = pd.DataFrame(scores)
        scores.index = topic_candidates.index.copy()
        self.eval_scores = scores
        return scores

    def save_results(self, directory=None, topics=True, scores=True, stats=True):
        if directory is None:
            directory = join(LDA_PATH, self.version, 'topics')
        if not exists(directory):
            makedirs(directory)
        model_name = self.dataset
        file_path = join(directory, model_name)

        if topics and self.topic_candidates is not None:
            fcsv = f'{file_path}_topic-candidates.csv'
            print(f"Writing topic candidates to {fcsv}")
            self.topic_candidates.to_csv(fcsv)

        if stats:
            fjson = f'{file_path}_reranker-statistics.json'
            with open(fjson, 'w') as fp:
                print(f"Writing Reranker statistics to {fjson}")
                json.dump(self.reranking_statistics(), fp, ensure_ascii=False, indent=2)

        if scores and self.eval_scores is not None:
            self.save_scores(self.eval_scores, model_name, directory)

    def plot(self):
        Reranker.plot_scores(self.eval_scores)
        self.scores_vec.boxplot()

    @classmethod
    def save_scores(cls, scores, dataset, directory=None):
        if directory is None:
            directory = join(LDA_PATH, 'topics')
        filename = join(directory, dataset)
        fcsv = f'{filename}_topic-scores.csv'
        print(f"Writing evaluation scores to {fcsv}")
        scores.to_csv(fcsv)

    @classmethod
    def _load(cls, path):
        print('Loading', path)
        return pd.read_csv(path, index_col=[0, 1, 2, 3, 4])

    @classmethod
    def load_topics_and_scores(cls, dataset, directory=None, joined=False, topics_only=False):
        if directory is None:
            directory = join(LDA_PATH, 'topics')
        topics = cls._load(join(directory, f'{dataset}_topic-candidates.csv'))
        if topics_only:
            return topics
        scores = cls._load(join(directory, f'{dataset}_topic-scores.csv'))
        if joined:
            return topics.join(scores)
        else:
            return topics, scores

    @staticmethod
    def plot_scores(scores):
        scores = scores.unstack('metric')
        for column in scores.columns.levels[0]:
            scores[column].reset_index(drop=True).plot(title=column, grid=True)
            descr = scores[column].describe()
            mean = descr.loc['mean']
            bestidx = mean.idxmax()
            bestval = mean[bestidx]
            print(f'reranking metric with highest score: {bestidx} [{bestval:.3f}]')
            print(descr.T[['mean', 'std']].sort_values('mean', ascending=False))
            print('-' * 50)


# --------------------------------------------------------------------------------------------------
# --- App ---


SAVE = True
PLOT = True
TOPN = 20
CORES = 4


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--version", type=str, required=False, default='noun')

    parser.add_argument("--topn", type=int, required=False, default=TOPN)
    parser.add_argument("--cores", type=int, required=False, default=CORES)
    parser.add_argument('--save', dest='save', action='store_true', required=False)
    parser.add_argument('--no-save', dest='save', action='store_false', required=False)
    parser.set_defaults(save=SAVE)
    parser.add_argument('--plot', dest='save', action='store_true', required=False)
    parser.add_argument('--no-plot', dest='save', action='store_false', required=False)
    parser.set_defaults(plot=PLOT)

    parser.add_argument("--metrics", nargs='*', type=str, required=False,
                        default=METRICS)
    parser.add_argument("--params", nargs='*', type=str, required=False,
                        default=PARAMS)
    parser.add_argument("--nbtopics", nargs='*', type=int, required=False,
                        default=NBTOPICS)

    args = parser.parse_args()
    return args


def rerank(
        dataset, version=None,
        topn=TOPN, save=SAVE, plot=PLOT, cores=CORES,
        metrics=METRICS, param_ids=PARAMS, nbs_topics=NBTOPICS
):
    topics_loader = TopicsLoader(
        dataset=dataset,
        param_ids=param_ids,
        nbs_topics=nbs_topics,
        version=version,
        topn=topn,
        include_corpus=True,
        include_texts=True
    )
    reranker = Reranker(topics_loader, processes=cores)
    reranker.rerank_coherence(metrics)
    reranker.rerank_vec()
    # tprint(reranker.topic_candidates)
    reranker.evaluate(reranker.topic_candidates)
    if save:
        reranker.save_results()
    if plot:
        reranker.plot()

    return reranker


def main():
    args = parse_args()
    print(pformat(vars(args)))

    reranker = rerank(
        dataset=DATASETS.get(args.dataset, args.dataset),
        version=args.version,
        topn=args.topn,
        param_ids=args.params,
        nbs_topics=args.nbtopics,
        metrics=args.metrics,
        save=args.save,
        plot=args.plot,
        cores=args.cores
    )
    print(reranker.eval_scores)


if __name__ == '__main__':
    main()
