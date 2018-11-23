# coding: utf-8
import argparse
import json
from os import makedirs
from os.path import join, exists
from time import time
import re

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import CoherenceModel, LdaModel
from pandas.core.common import SettingWithCopyWarning

from constants import ETL_PATH, BAD_TOKENS, DATASETS, METRICS, PARAMS, NBTOPICS
import warnings
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

pd.options.display.precision = 3
np.set_printoptions(precision=3, threshold=None, edgeitems=None, linewidth=800, suppress=None)

PLACEHOLDER = '[[PLACEHOLDER]]'


# --------------------------------------------------------------------------------------------------
# --- TopicLoader Class ---


class TopicsLoader(object):

    def __init__(self, dataset, param_ids: list, nbs_topics: list,
                 version=None, nbfiles=None, corpus_type='bow', topn=20):
        self.topn = topn
        self.dataset = dataset
        self.version = version
        self.param_ids = param_ids
        self.nb_topics_list = nbs_topics
        self.nb_topics = sum(nbs_topics) * len(param_ids)
        self.corpus_type = corpus_type
        self.directory = join(ETL_PATH, 'LDAmodel', self.version)
        self.nbfiles = nbfiles
        self.nbfiles_str = f'_nbfiles{nbfiles:02d}' if nbfiles else ''
        self.data_filename = f'{dataset}{self.nbfiles_str}_{version}_{self.corpus_type}'
        self.pat = re.compile(r'^([0-9]+.*?)*?[A-Za-zÄÖÜäöü].*')
        self.dict_from_corpus = self._load_dict()
        self.corpus = self._load_corpus()
        self.texts = self._load_texts()
        self.topics = self._topn_topics()

    def _topn_topics(self):
        """
        get the topn topics from the LDA-model in DataFrame format
        """
        all_topics = []
        for param_id in self.param_ids:
            for nb_topics in self.nb_topics_list:
                ldamodel = self._load_model(param_id, nb_topics)
                # topic building ignoring placeholder values
                topics = []
                for i in range(nb_topics):
                    topic = []
                    for term in ldamodel.get_topic_terms(i, topn=self.topn+10):
                        token = ldamodel.id2word[term[0]]
                        if token not in BAD_TOKENS and self.pat.match(token):
                            topic.append(token)
                            if len(topic) == self.topn:
                                break
                        else:
                            # print(token)
                            pass
                    topics.append(topic)

                model_topics = (
                    pd.DataFrame(topics, columns=['term' + str(i) for i in range(self.topn)])
                    .assign(
                        dataset=f'{self.dataset}{self.nbfiles}' if self.nbfiles else self.dataset,
                        param_id=param_id,
                        nb_topics=nb_topics
                    )
                )
                all_topics.append(model_topics)
        topics = (
            pd.concat(all_topics)
            .rename_axis('topic_idx')
            .reset_index(drop=False)
            .set_index(['dataset', 'param_id', 'nb_topics', 'topic_idx'])
        )
        return topics

    def topic_ids(self):
        return self.topics.applymap(lambda x: self.dict_from_corpus.token2id[x])

    def _load_model(self, param_id, nb_topics):
        """
        Load an LDA model.
        """
        model_filename = f'{self.dataset}{self.nbfiles_str}_LDAmodel_{param_id}_{nb_topics}'
        path = join(self.directory, param_id, model_filename)
        print('Loading model from', path)
        ldamodel = LdaModel.load(path)
        return ldamodel

    def _load_dict(self):
        """
        This dictionary is a different from the model's dict with a different word<->id mapping,
        but from the same corpus and will be used for the Coherence Metrics.
        """
        dict_path = join(self.directory, self.data_filename + '.dict')
        print('loading dictionary from', dict_path)
        dict_from_corpus: Dictionary = Dictionary.load(dict_path)
        dict_from_corpus.add_documents([[PLACEHOLDER]])
        _ = dict_from_corpus[0]  # init dictionary
        return dict_from_corpus

    def _load_corpus(self):
        """
        load corpus (for u_mass scores)
        """
        corpus_path = join(self.directory, self.data_filename + '.mm')
        print('loading corpus from', corpus_path)
        corpus = MmCorpus(corpus_path)
        corpus = list(corpus)
        corpus.append([(self.dict_from_corpus.token2id[PLACEHOLDER], 1.0)])
        return corpus

    def _load_texts(self):
        """
        load texts (for c_... scores using sliding window)
        """
        doc_path = join(self.directory, self.data_filename + '_texts.json')
        with open(doc_path, 'r') as fp:
            print('loading texts from', doc_path)
            texts = json.load(fp)
        texts.append([PLACEHOLDER])
        return texts


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
        self.dict_from_corpus = topics.dict_from_corpus
        self.placeholder_id = topics.dict_from_corpus.token2id[PLACEHOLDER]
        self.corpus = topics.corpus
        self.texts = topics.texts
        self.nb_topics = topics.nb_topics
        self.topic_terms = topics.topics
        self.topic_ids = topics.topic_ids()
        self.nb_top_terms = nb_top_terms
        self.processes = processes
        self.dataset = topics.dataset
        self.version = topics.version
        self.nbfiles_str = topics.nbfiles_str

        if nb_candidate_terms is None:
            self.nb_candidate_terms = topics.topn
        else:
            self.nb_candidate_terms = nb_candidate_terms

        # this method is only needed for the fast rerank algorithm
        # once other algorithms are implemented it should be removed from the constructor
        self.shifted_topics = self._shift_topics()
        self.topic_candidates = None
        self.eval_scores = None

        # generate some statistics
        self._statistics_ = dict()
        self._statistics_['dataset'] = topics.dataset
        self._statistics_['version'] = topics.version
        self._statistics_['nbfiles'] = topics.nbfiles

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

    def _id2term(self, id_):
        return self.dict_from_corpus[id_]

    def _vote(self, group):
        """
        This function returns a topic generated from majority votes of the other ranking methods.
        :param group: applied on a DataFrame topic group
        :return:
        """
        # count terms and remove placeholder
        y = (
            group
            .apply(pd.value_counts)
            .sum(axis=1)
            .astype(np.int16)
            .sort_values(ascending=False, kind='mergesort')
        )
        y = y[y.index != self.placeholder_id]

        # this is a bit delicate for terms with the same (min) count
        # which may or may not be in the final set. Therefore we address this case separately
        if len(y) > self.nb_top_terms:
            min_vote = y.iloc[self.nb_top_terms - 1]
            min_vote2 = y.iloc[self.nb_top_terms]
            split_on_min_vote = (min_vote == min_vote2)
        else:
            min_vote = 0
            split_on_min_vote = False

        df = y.to_frame(name='counter')
        # restore original order
        topic_id = group.index.get_level_values('topic_idx')[0]
        reference_order = pd.Index(self.topic_ids.iloc[topic_id])
        df['ref_idx'] = y.index.map(reference_order.get_loc)

        if split_on_min_vote:
            nb_above = (y > min_vote).sum()
            remaining_places = self.nb_top_terms - nb_above
            df_min = df[df.counter == min_vote]
            df_min = df_min.sort_values('ref_idx')
            df_min = df_min[:remaining_places]
            df_above = df[df.counter > min_vote]
            df = df_above.append(df_min)
        else:
            df = df[df.counter >= min_vote]

        df = (
            df
            .sort_values('ref_idx')
            .reset_index(drop=False)
            .rename(columns={'index': 'ids'})
            .drop('counter', axis=1)
            .T
        )
        return df

    def oop_score(self, top_scores):
        """
        Avg. out of place score per metric in relation to the reference topics.
        Higher scores indicate a stronger degree of reranking while lower scores
        indicate a stronger similarity with the reference topics.

        :param top_scores: numpy array with the reranked indices of the candidate terms.
                           generated from a metric. shape: (nb_top_terms, nb_topics)
        :return:           float (scalar) of mean of a oop score over all given topics.
        """
        refgrid = np.mgrid[0:self.nb_topics, 0:self.nb_top_terms][1]
        oop = np.abs(top_scores - refgrid).sum() / self.nb_topics
        print(f'    avg out of place score compared to original topic term rankings: {oop:.1f}')
        return oop

    def get_reference(self):
        metric = 'ref'
        ref_topics_terms = (
            self.topic_ids.iloc[:, :self.nb_top_terms]
            .copy()
            .reset_index(drop=True)
            .assign(metric=metric)
        )
        self._statistics_[metric] = dict()
        self._statistics_[metric]['oop_score'] = 0
        self._statistics_[metric]['runtime'] = 0
        return ref_topics_terms

    def rerank_by_vote(self, topic_candidates):
        t0 = time()
        metric = 'vote'
        print(f'Calculating topic candidates using majority vote '
              f'on {self.nb_candidate_terms} candidate terms '
              f'for {self.nb_topics} topics')

        topic_votes = (
            topic_candidates
            .sort_index(kind='mergesort')
            .rename_axis('topic_idx')
            .reset_index(drop=False)
            .set_index(['topic_idx', 'metric'])
            .groupby('topic_idx', sort=False).apply(self._vote)
        )
        oop_indices = topic_votes.xs('ref_idx', level=1, drop_level=False).values
        topic_votes = (
            topic_votes
            .xs('ids', level=1, drop_level=False)
            .reset_index(drop=True)
            .rename(columns=lambda x: topic_candidates.columns[x])
            .assign(metric=metric)
        )

        t1 = int(time() - t0)
        self._statistics_[metric] = dict()
        self._statistics_[metric]['oop_score'] = self.oop_score(oop_indices)
        self._statistics_[metric]['runtime'] = t1
        print("    done in {:02d}:{:02d}:{:02d}".format(t1 // 3600, (t1 // 60) % 60, t1 % 60))
        return topic_votes

    def rerank_fast_per_metric(self, metric, coherence_model=None):
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
            .from_records(tpx_ids, columns=self.topic_terms.columns[:self.nb_top_terms])
            .assign(metric=metric)
        )

        t1 = int(time() - t0)
        self._statistics_[metric] = dict()
        self._statistics_[metric]['oop_score'] = self.oop_score(top_scores)
        self._statistics_[metric]['runtime'] = t1
        print("    done in {:02d}:{:02d}:{:02d}".format(t1 // 3600, (t1 // 60) % 60, t1 % 60))
        return tpx_ids

    def rerank_fast(self, metrics=None):
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
            umass_topics_terms = self.rerank_fast_per_metric('u_mass')
            candidates.append(umass_topics_terms)
        if 'c_v' in metrics:
            cv_topics_terms = self.rerank_fast_per_metric('c_v')
            candidates.append(cv_topics_terms)
        if 'c_uci' in metrics:
            cuci_topics_terms = self.rerank_fast_per_metric('c_uci')
            candidates.append(cuci_topics_terms)
        if 'c_npmi' in metrics:
            cnpmi_topics_terms = self.rerank_fast_per_metric('c_npmi')
            candidates.append(cnpmi_topics_terms)
        topic_candidates = pd.concat(candidates, axis=0)

        # adding candidates by majority votes from prior reference and rerankings
        if 'vote' in metrics:
            vote_topic_terms = self.rerank_by_vote(topic_candidates)
            topic_candidates = topic_candidates.append(vote_topic_terms)

        topic_candidates = (
            topic_candidates
            .groupby('metric', sort=False, as_index=False, group_keys=False)
            .apply(lambda x: x.set_index(self.topic_terms.index))
            .set_index('metric', append=True)
            # replacing token-ids with tokens -> resulting in the final topic candidates
            .applymap(self._id2term)
            .reorder_levels([0, -1, 1, 2, 3])
        )
        self.topic_candidates = topic_candidates
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
            topic_candidates = self.topic_candidates.values.tolist()

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
            directory = join(ETL_PATH, 'LDAmodel', self.version, 'Reranker')
        if not exists(directory):
            makedirs(directory)
        model_name = self.dataset + self.nbfiles_str
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

    @classmethod
    def save_scores(cls, scores, dataset, directory=None):
        if directory is None:
            directory = join(ETL_PATH, 'LDAmodel', 'Reranker')
        filename = join(directory, dataset)
        fcsv = f'{filename}_evaluation-scores.csv'
        print(f"Writing evaluation scores to {fcsv}")
        scores.to_csv(fcsv)

    @classmethod
    def _load(cls, path):
        print('Loading', path)
        return pd.read_csv(path, index_col=[0, 1, 2, 3, 4])

    @classmethod
    def load_topics_and_scores(cls, dataset, directory=None, joined=False, topics_only=False):
        if directory is None:
            directory = join(ETL_PATH, 'LDAmodel', 'Reranker')
        topics = cls._load(join(directory, f'{dataset}_topic-candidates.csv'))
        if topics_only:
            return topics
        scores = cls._load(join(directory, f'{dataset}_evaluation-scores.csv'))
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
PLOT = False
TOPN = 20
CORES = 4


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--version", type=str, required=False, default='default')
    parser.add_argument("--nbfiles", type=int, required=False, default=None)

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
        dataset, version=None, nbfiles=None,
        topn=TOPN, save=SAVE, plot=PLOT, cores=CORES,
        metrics=METRICS, param_ids=PARAMS, nbs_topics=NBTOPICS
):
    topics_loader = TopicsLoader(
        dataset=dataset,
        param_ids=param_ids,
        nbs_topics=nbs_topics,
        version=version,
        nbfiles=nbfiles,
        topn=topn
    )
    reranker = Reranker(topics_loader, processes=cores)
    topic_candidates = reranker.rerank_fast(metrics)
    reranker.evaluate(topic_candidates)
    if save:
        reranker.save_results()
    if plot:
        reranker.plot()

    return reranker


def main():
    args = parse_args()
    print(args)

    reranker = rerank(
        dataset=DATASETS.get(args.dataset, args.dataset),
        version=args.version,
        nbfiles=args.nbfiles,
        topn=args.topn,
        param_ids=args.params,
        nbs_topics=args.nbtopics,
        metrics=args.metrics,
        save=args.save,
        plot=args.plot,
        cores=args.cores
    )
    topics = reranker.topic_candidates
    scores = reranker.eval_scores


if __name__ == '__main__':
    main()
