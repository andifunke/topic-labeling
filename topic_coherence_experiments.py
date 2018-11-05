# coding: utf-8

import json
from itertools import chain
from os.path import join
from pprint import pprint
from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import CoherenceModel, LdaModel
from pandas.core.common import SettingWithCopyWarning

from constants import (
    ETL_PATH
)
import warnings
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
from utils import tprint

pd.options.display.max_rows = 2001
pd.options.display.precision = 3
np.set_printoptions(precision=3, threshold=None, edgeitems=None, linewidth=800, suppress=None)

datasets = {
    'E': 'Europarl',
    'FA': 'FAZ_combined',
    'FO': 'FOCUS_cleansed',
    'O': 'OnlineParticipation',
    'P': 'PoliticalSpeeches',
    'dewi': 'dewiki',
    'dewa': 'dewac',
}
bad_tokens = {
    'Europarl': [
        'E.', 'Kerr', 'The', 'la', 'ia', 'For', 'Ieke', 'the',
    ],
    'FAZ_combined': [
        'S.', 'j.reinecke@faz.de', 'B.',
    ],
    'FOCUS_cleansed': [],
    'OnlineParticipation': [
        'Re', '@#1', '@#2', '@#3', '@#4', '@#5', '@#6', '@#7', '@#8', '@#9', '@#1.1', 'Für', 'Muss',
        'etc', 'sorry', 'Ggf', 'u.a.', 'z.B.', 'B.', 'stimmt', ';-)', 'lieber', 'o.', 'Ja',
        'Desweiteren',
    ],
    'PoliticalSpeeches': [],
    'dewiki': [],
    'dewac': [],
}
all_bad_tokens = set(chain(*bad_tokens.values()))
params_list = ['a42', 'b42', 'c42', 'd42', 'e42']
placeholder = '[[PLACEHOLDER]]'


# --------------------------------------------------------------------------------------------------
# --- TopicLoader Class ---


class TopicsLoader(object):

    def __init__(self, dataset_name, param_ids: list, nbs_topics: list, topn=20):
        self.topn = topn
        self.dataset = dataset_name
        self.param_ids = param_ids
        self.nb_topics_list = nbs_topics
        self.nb_topics = sum(nbs_topics) * len(param_ids)
        split_type = 'fullset'
        self.corpus_type = 'bow'
        self.data_filename = f'{dataset_name}_{split_type}_nouns_{self.corpus_type}'
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
                # topics = [
                #     [ldamodel.id2word[term[0]] for term in ldamodel.get_topic_terms(i, topn=self.topn)]
                #     for i in range(nb_topics)
                # ]

                # alternative topic building ignoring placeholder values
                topics = []
                for i in range(nb_topics):
                    topic = []
                    for term in ldamodel.get_topic_terms(i, topn=self.topn+10):
                        token = ldamodel.id2word[term[0]]
                        if token not in all_bad_tokens:
                            topic.append(token)
                            if len(topic) == self.topn:
                                break
                    topics.append(topic)

                model_topics = (
                    pd.DataFrame(topics, columns=['term' + str(i) for i in range(self.topn)])
                    # .applymap(lambda x: placeholder if x in all_bad_tokens else x)
                    .assign(
                        dataset=self.dataset,
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
        model_filename = f'{self.dataset}_LDAmodel_{param_id}_{nb_topics}'
        path = join(ETL_PATH, 'LDAmodel', param_id, model_filename)
        print('loading model from', path)
        return LdaModel.load(path)

    def _load_dict(self):
        """
        This dictionary is a different from the model's dict with a different word<->id mapping,
        but from the same corpus and will be used for the Coherence Metrics.
        """
        dict_path = join(ETL_PATH, 'LDAmodel', self.data_filename + '.dict')
        print('loading dictionary from', dict_path)
        dict_from_corpus: Dictionary = Dictionary.load(dict_path)
        dict_from_corpus.add_documents([[placeholder]])
        _ = dict_from_corpus[0]  # init dictionary
        return dict_from_corpus

    def _load_corpus(self):
        """
        load corpus (for u_mass scores)
        """
        corpus_path = join(ETL_PATH, 'LDAmodel', self.data_filename + '.mm')
        print('loading corpus from', corpus_path)
        corpus = MmCorpus(corpus_path)
        corpus = list(corpus)
        corpus.append([(self.dict_from_corpus.token2id[placeholder], 1.0)])
        return corpus

    def _load_texts(self):
        """
        load texts (for c_... scores using sliding window)
        """
        doc_path = join(
            ETL_PATH, 'LDAmodel',
            self.data_filename.rstrip(f'{self.corpus_type}') + 'texts.json'
        )
        with open(doc_path, 'r') as fp:
            print('loading texts from', doc_path)
            texts = json.load(fp)
        texts.append([placeholder])
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
        self.placeholder_id = topics.dict_from_corpus.token2id[placeholder]
        self.corpus = topics.corpus
        self.texts = topics.texts
        self.nb_topics = topics.nb_topics
        self.topic_terms = topics.topics
        self.topic_ids = topics.topic_ids()
        self.nb_top_terms = nb_top_terms
        self.processes = processes
        self.dataset = topics.dataset

        if nb_candidate_terms is None:
            self.nb_candidate_terms = topics.topn
        else:
            self.nb_candidate_terms = nb_candidate_terms

        self.shifted_topics = self._shift_topics()
        self.topic_candidates = None
        self._statistics_ = dict()
        self._statistics_['dataset'] = topics.dataset

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
        print(f'avg out of place score compared to original topic term rankings: {oop:.1f}')
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
        print(f'producing coherence scores for {metric} measure '
              f'using {self.nb_candidate_terms} candidate terms '
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
        print("done in {:02d}:{:02d}:{:02d}".format(t1 // 3600, (t1 // 60) % 60, t1 % 60))
        return tpx_ids, cm

    def rerank_by_vote(self, topic_candidates):
        t0 = time()
        metric = 'vote'
        print(f'producing candidates via majority vote '
              f'using {self.nb_candidate_terms} candidate terms '
              f'for {len(topic_candidates)} topics')

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
        print("done in {:02d}:{:02d}:{:02d}".format(t1 // 3600, (t1 // 60) % 60, t1 % 60))
        return topic_votes

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
        available_metrics = ['u_mass', 'c_v', 'c_uci', 'c_npmi', 'vote']
        if metrics is None:
            metrics = available_metrics

        print(f'creating reranked top candidates for metrics {metrics}, using fast method')
        candidates = []

        # adding original (reference) topics
        ref_topics_terms = self.get_reference()
        candidates.append(ref_topics_terms)

        # adding several rerankings according to different metrics
        cm = None
        if 'u_mass' in metrics:
            umass_topics_terms, _ = self.rerank_fast_per_metric('u_mass')
            candidates.append(umass_topics_terms)
        if 'c_v' in metrics:
            cv_topics_terms, _ = self.rerank_fast_per_metric('c_v')
            candidates.append(cv_topics_terms)
        if 'c_uci' in metrics:
            cuci_topics_terms, cm = self.rerank_fast_per_metric('c_uci', cm)
            candidates.append(cuci_topics_terms)
        if 'c_npmi' in metrics:
            cnpmi_topics_terms, cm = self.rerank_fast_per_metric('c_npmi', cm)
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
        )
        self.topic_candidates = topic_candidates
        return topic_candidates

    def rerank_greedy(self):
        pass

    def rerank_full(self):
        pass

    def evaluate(self, topic_candidates=None, nbtopterms=None):
        print('evaluating topic candidates')

        # reference scores per topic for top topic terms
        if nbtopterms is None:
            nbtopterms = self.nb_top_terms

        if topic_candidates is None:
            topic_candidates = self.topic_candidates

        cm_umass = CoherenceModel(
            topics=topic_candidates, corpus=self.corpus, dictionary=self.dict_from_corpus,
            coherence='u_mass', topn=nbtopterms, processes=self.processes
        )
        umass_scores = cm_umass.get_coherence_per_topic(with_std=False, with_support=False)

        cm_cv = CoherenceModel(
            topics=topic_candidates, texts=self.texts, dictionary=self.dict_from_corpus,
            coherence='c_v', topn=nbtopterms, processes=self.processes
        )
        cv_scores = cm_cv.get_coherence_per_topic()

        # changed segmentation for c_uci and c_npmi from s_one_set to s_one_one (default)
        cm_cuci = CoherenceModel(
            topics=topic_candidates, texts=self.texts, dictionary=self.dict_from_corpus,
            coherence='c_uci', topn=nbtopterms, processes=self.processes
        )
        cuci_scores = cm_cuci.get_coherence_per_topic()

        cm_cuci.coherence = 'c_npmi'  # reusing precalculated probability estimates
        cnpmi_scores = cm_cuci.get_coherence_per_topic()

        scores = {
            'u_mass': umass_scores,
            'c_v': cv_scores,
            'c_uci': cuci_scores,
            'c_npmi': cnpmi_scores,
        }
        return scores

    def stats(self, scores: list, columns: list, nbtopics: int = None) -> pd.DataFrame:
        """
        :rtype: pd.DataFrame
        """
        if nbtopics is None:
            nbtopics = self.nb_topics

        length = len(columns)
        df = pd.DataFrame(np.reshape(scores, (length, nbtopics)), index=columns).T
        df.plot()
        plt.show()
        descr = df.describe()
        mean = descr.loc['mean']
        bestidx = mean.idxmax()
        bestval = mean[bestidx]
        print(f'topic reranking with highest score: {bestidx} [{bestval:.3f}]')
        print(descr.T[['mean', 'std']].sort_values('mean', ascending=False))
        return descr

    def statistics(self):
        self._statistics_['nb_topics'] = self.nb_topics
        self._statistics_['nb_candidate_terms'] = self.nb_candidate_terms
        self._statistics_['nb_top_terms'] = self.nb_top_terms
        self._statistics_['size_vocabulary'] = len(self.dict_from_corpus)
        self._statistics_['size_corpus'] = len(self.corpus)
        return self._statistics_

    def save_results(self, directory):
        filename = join(directory, self.dataset)

        if self.topic_candidates is not None:
            fcsv = f'{filename}_topic-candidates.csv'
            print(f"writing topic candidates fo {fcsv}")
            self.topic_candidates.to_csv(fcsv)

        fjson = f'{filename}_reranker-statistics.json'
        with open(fjson, 'w') as fp:
            print(f"writing Reranker statistics to {fjson}")
            json.dump(self.statistics(), fp, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------------------------------
# --- App ---


def rerank(dataset, param_ids, nbs_topics, topns=None, metrics=None, evaluate=False, save=True):
    if topns is None:
        topns = [20]
    if metrics is None:
        metrics = ['u_mass', 'c_v', 'c_uci', 'c_npmi', 'vote']

    reranker = None
    all_topic_candidates = []
    for n in topns:
        topics_loader = TopicsLoader(dataset, param_ids, nbs_topics, topn=n)
        reranker = Reranker(topics_loader, nb_candidate_terms=n, processes=4)
        reranker.rerank_fast(metrics)
        if save:
            reranker.save_results(join(ETL_PATH, 'Reranker'))
        all_topic_candidates.append(reranker)

    all_topic_candidates = pd.concat(all_topic_candidates)

    if evaluate:
        all_tc_list = all_topic_candidates.drop('metric', axis=1).values.tolist()
        scores = reranker.evaluate(all_tc_list)

        columns = ['ref'] + metrics
        columns = ['_'.join([s, str(n)]) for n in topns for s in columns]
        stats_umass = reranker.stats(scores['u_mass'], columns).assign(eval_metric='u_mass')
        stats_cv = reranker.stats(scores['c_v'], columns).assign(eval_metric='c_v')
        stats_cuci = reranker.stats(scores['c_uci'], columns).assign(eval_metric='c_uci')
        stats_cnpmi = reranker.stats(scores['c_npmi'], columns).assign(eval_metric='c_npmi')
        stats = (
            pd.concat([stats_umass, stats_cv, stats_cuci, stats_cnpmi])
            .assign(
                dataset=dataset,
                # nbtopics=nbtopics,
                # param=param
            )
        )
    else:
        stats = None

    return all_topic_candidates, stats


def main():
    param_ids = ['a42', 'b42', 'c42', 'd42', 'e42']
    nbs_topics = [10, 25, 50, 100]

    topics, stats = rerank(
        datasets['O'],
        param_ids[:],
        nbs_topics[:],
        # metrics=['u_mass', 'vote'],
        save=True,
    )
    tprint(topics, 0)


if __name__ == '__main__':
    main()


# from a tutorial:
# ##### c_uci and c_npmi coherence measures
# c_v and c_uci and c_npmi all use the boolean sliding window approach of estimating probabilities.
# Since the `CoherenceModel` caches the accumulated statistics, calculation of c_uci and c_npmi are
# practically free after calculating c_v coherence. These two methods are simpler and were shown to
# correlate less with human judgements than c_v but more so than u_mass.
#
# ^ **this is not correct. c_uci and c_npmi use per default a different segmentation than c_v**
