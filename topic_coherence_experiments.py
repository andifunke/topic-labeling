# coding: utf-8

import json
from itertools import chain
from os.path import join

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


class ModelLoader(object):

    def __init__(self, dataset_name, param_id, nbtopics):
        self.dataset = dataset_name
        self.param = param_id
        self.nbtopics = nbtopics
        self.ldamodel = self._load_model()
        self.dict_from_model = self.ldamodel.id2word
        split_type = 'fullset'
        self.corpus_type = 'bow'
        self.data_filename = f'{dataset_name}_{split_type}_nouns_{self.corpus_type}'
        self.dict_from_corpus = self._load_dict()
        self.corpus = self._load_corpus()
        self.texts = self._load_texts()

    def _load_model(self):
        """
        Load an LDA model.
        """
        model_filename = f'{self.dataset}_LDAmodel_{self.param}_{self.nbtopics}'
        path = join(ETL_PATH, 'LDAmodel', self.param, model_filename)
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
        # this!
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
        # this!
        texts.append([placeholder])
        return texts


class Reranker(object):

    def __init__(self, modelloader, topn=20, nbtopterms=10, processes=-1):
        """
        :param modelloader: pass instance of ModelLoader
        :param topn:        number to topic terms to evaluate the model over. topn must be > nbtopterms.
        :param nbtopterms:  number of remaining topic terms. The size of the final topic representation
                            set. nbtopterms ust be < topn.
        :param processes:   number of workers / CPUs used for calculation
        """
        self.dict_from_corpus = modelloader.dict_from_corpus
        self.corpus = modelloader.corpus
        self.texts = modelloader.texts
        self.nbtopics = modelloader.nbtopics
        self.nbtopterms = nbtopterms
        self.topn = topn
        self.processes = processes

        self.topic_terms = self._topn_topics(modelloader)
        self.topic_terms_ids = self._topn_topics_ids()
        self.shifted_topics = self._shift_topn_topics()
        self.topic_candidates = None

    def _topn_topics(self, modelloader):
        """
        get the topn topics from the LDA-model in DataFrame format
        :type modelloader: ModelLoader
        """
        ldamodel = modelloader.ldamodel
        dict_from_model = modelloader.dict_from_model
        dataset = modelloader.dataset

        topics = [
            [dataset] +
            [dict_from_model[term[0]] for term in ldamodel.get_topic_terms(i, topn=self.topn)]
            for i in range(self.nbtopics)
        ]
        df_topics = pd.DataFrame(
            topics,
            columns=['dataset'] + ['term' + str(i) for i in range(self.topn)]
        )
        df_topics = df_topics.applymap(lambda x: placeholder if x in all_bad_tokens else x)
        topic_terms = df_topics.iloc[:, 1:]
        return topic_terms

    def _topn_topics_ids(self):
        topic_terms_ids = self.topic_terms.applymap(lambda x: self.dict_from_corpus.token2id[x])
        return topic_terms_ids

    def _shift_topn_topics(self):
        """
        from the top n terms construct all topic set that omit one term,
        resulting in n topics with n-1 topic terms for each topic
        """
        shifted_frames = []
        for i in range(self.topn):
            df = pd.DataFrame(np.roll(self.topic_terms_ids.values, shift=-i, axis=1))
            # tprint(df.applymap(lambda x: self.dict_from_corpus[x]))
            shifted_frames.append(df)
        shifted_ids = pd.concat(shifted_frames)
        # omit the first topic term, then the second and append the first etc...
        shifted_topics = shifted_ids.iloc[:, 1:].values.tolist()
        # print('topics shape: %s' % str(np.asarray(shifted_topics).shape))
        return shifted_topics

    def _vote(self, group):
        """
        This function returns a topic generated from majority votes of the other ranking methods.
        :param group: applied on a DataFrame topic group
        :return:
        """
        # count terms and remove placeholder
        y = group.apply(pd.value_counts).sum(axis=1).astype(np.int16).sort_values(ascending=False)
        y = y[y.index != placeholder]

        # restore original order
        reference_order = pd.Index(group[group.index.get_level_values('source') == 'ref'].squeeze())
        # this is a bit delicate for terms with the same (min) count
        # which may or may not be in the final set. Therefore we address this case separately
        if len(y) > self.nbtopterms:
            min_vote = y[self.nbtopterms-1]
            min_vote2 = y[self.nbtopterms]
            split_on_min_vote = (min_vote == min_vote2)
        else:
            min_vote = 0
            split_on_min_vote = False

        def indices(x):
            if x in reference_order:
                idx = reference_order.get_loc(x)
            else:
                idx = self.nbtopterms
            return idx

        df = y.to_frame(name='counter')
        df['idx'] = y.index.map(indices)

        if split_on_min_vote:
            nb_above = (y > min_vote).sum()
            remaining_places = self.nbtopterms - nb_above
            df_min = df[df.counter == min_vote]
            df_min = df_min.sort_values('idx')
            df_min = df_min[:remaining_places]
            df_above = df[df.counter > min_vote]
            df = df_above.append(df_min)
        else:
            df = df[df.counter >= min_vote]

        df = df.sort_values('idx')
        return pd.Series(df.index)

    def _id2term(self, id_):
        return self.dict_from_corpus[id_]

    def rerank_fast_per_metric(self, measure, coherence_model=None):
        if self.shifted_topics is None:
            self.shifted_topics = self._shift_topn_topics()
        # print('shifted topics shape', np.asarray(self.shifted_topics).shape)

        print(f'produce scores for {measure}, n={self.topn}')
        # calculate the scores for all shifted topics
        kwargs = dict(
            topics=self.shifted_topics,
            dictionary=self.dict_from_corpus,
            coherence=measure,
            topn=self.topn-1,
            processes=self.processes
        )
        if measure == 'u_mass':
            kwargs['corpus'] = self.corpus
        else:
            kwargs['texts'] = self.texts

        if coherence_model is None:
            cm = CoherenceModel(**kwargs)
        else:
            cm = coherence_model
            cm.coherence = measure

        scores1d = cm.get_coherence_per_topic()
        # print('len scores1d', len(scores1d))
        # print(np.ascarray(scores1d))
        scores2d = np.reshape(scores1d, (self.topn, -1)).T
        # print('shape scores2d', scores2d.shape)
        # print(scores2d)

        # the highest values indicate the terms whose absence improves the topic coherence most
        sorted_scores = np.argsort(scores2d, axis=1)
        # print('sorted scores', sorted_scores.shape)
        # print(np.sort(scores2d, axis=1))
        # print(sorted_scores)
        # thus we will keep the first nbtopterms (default 10) indices
        top_scores = sorted_scores[:, :self.nbtopterms]
        # print('top scores', top_scores.shape)
        # print(top_scores)
        # and sort them back for convenience
        top_scores = np.sort(top_scores, axis=1)
        # print('top scores sorted', top_scores.shape)
        # print(top_scores)

        # out of place score compared to the reference topics
        refgrid = np.mgrid[0:self.nbtopics, 0:self.nbtopterms][1]
        oop_score = np.abs(top_scores - refgrid).sum() / self.nbtopics
        print(f'avg out of place score compared to original topic terms: {oop_score}')

        # replacing indices with token-ids
        tpx_ids = []
        for i in range(self.nbtopics):
            tpx = self.topic_terms_ids.values[i, top_scores[i]]
            tpx_ids.append(tpx)
        tpx_ids = pd.DataFrame.from_records(tpx_ids, columns=self.topic_terms.columns[:self.nbtopterms])
        # print('inserted term ids', tpx_ids.shape)
        # tprint(tpx_ids, 0)

        # replacing token-ids with tokens -> resulting in the final topic candidates
        tpx_terms = tpx_ids.applymap(self._id2term)
        # print('inserted term ids', tpx_terms.shape)
        # tprint(tpx_terms, 0)

        return tpx_terms, cm

    def rerank_fast(self, metrics=None):
        """
        This is the main method for a Reranker to call.
        :param   metrics -> list of str. str must be in {'u_mass', 'c_v', 'c_uci', 'c_npmi'}
        :return:
        """
        available_metrics = ['u_mass', 'c_v', 'c_uci', 'c_npmi', 'vote']
        if metrics is None:
            metrics = available_metrics

        candidates = []

        print(f'creating reranked topc candidates for metrics {metrics}, using fast method')

        ref_topics_terms = self.topic_terms.iloc[:, :self.nbtopterms]
        ref_topics_terms['source'] = f'ref_{self.topn}'
        candidates.append(ref_topics_terms)

        cm = None
        if 'u_mass' in metrics:
            umass_topics_terms, _ = self.rerank_fast_per_metric('u_mass')
            umass_topics_terms['source'] = f'umass_{self.topn}'
            candidates.append(umass_topics_terms)
        if 'c_v' in metrics:
            cv_topics_terms, _ = self.rerank_fast_per_metric('c_v')
            cv_topics_terms['source'] = f'cv_{self.topn}'
            candidates.append(cv_topics_terms)
        if 'c_uci' in metrics:
            cuci_topics_terms, cm = self.rerank_fast_per_metric('c_uci', cm)
            cuci_topics_terms['source'] = f'cuci_{self.topn}'
            candidates.append(cuci_topics_terms)
        if 'c_npmi' in metrics:
            cnpmi_topics_terms, cm = self.rerank_fast_per_metric('c_npmi', cm)
            cnpmi_topics_terms['source'] = f'cnpmi_{self.topn}'
            candidates.append(cnpmi_topics_terms)

        # combining the topics candidates
        topic_candidates = pd.concat(candidates, axis=0)

        # adding candidates by majority votes
        if 'vote' in metrics:
            tc_grouped = (
                topic_candidates
                .sort_index(kind='mergesort')
                .assign(topic=lambda x: x.index)
                .set_index(['topic', 'source'])
            )
            topic_votes = tc_grouped.groupby('topic').apply(self._vote)
            topic_votes = topic_votes.rename(columns=lambda x: topic_candidates.columns[x])
            topic_votes['source'] = f'vote_{self.topn}'
            topic_candidates = topic_candidates.append(topic_votes)

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
            nbtopterms = self.nbtopterms

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
            nbtopics = self.nbtopics

        length = len(columns)
        df = pd.DataFrame(np.reshape(scores, (length, nbtopics)), index=columns).T
        df.plot()
        plt.show()
        descr = df.describe()
        mean = descr.loc['mean']
        bestidx = mean.idxmax()
        bestval = mean[bestidx]
        print(f'topic reranking with highest score: {bestidx} [{round(bestval, 3)}]')
        print(descr.T[['mean', 'std']].sort_values('mean', ascending=False))
        return descr


def rerank(dataset, nbtopics, param, topns=None, metrics=None):
    model_loader = ModelLoader(dataset, param, nbtopics)
    if topns is None:
        topns = [20]  # [20, 30, 50]
    if metrics is None:
        metrics = ['u_mass', 'c_v', 'c_uci', 'c_npmi', 'vote']
    all_topic_candidates = []
    reranker = None
    for n in topns:
        reranker = Reranker(model_loader, topn=n, processes=4)
        tc = reranker.rerank_fast(metrics)
        all_topic_candidates.append(tc)

    all_topic_candidates = pd.concat(all_topic_candidates)
    columns = ['ref'] + metrics
    columns = ['_'.join([s, str(n)]) for n in topns for s in columns]
    all_tc_list = all_topic_candidates.drop('source', axis=1).values.tolist()
    scores = reranker.evaluate(all_tc_list)

    stats_umass = reranker.stats(scores['u_mass'], columns).assign(eval_metric='u_mass')
    stats_cv = reranker.stats(scores['c_v'], columns).assign(eval_metric='c_v')
    stats_cuci = reranker.stats(scores['c_uci'], columns).assign(eval_metric='c_uci')
    stats_cnpmi = reranker.stats(scores['c_npmi'], columns).assign(eval_metric='c_npmi')
    stats = (
        pd.concat([stats_umass, stats_cv, stats_cuci, stats_cnpmi])
        .assign(
            dataset=dataset,
            nbtopics=nbtopics,
            param=param
        )
    )
    return stats


# from a tutorial
# ##### c_uci and c_npmi coherence measures
# c_v and c_uci and c_npmi all use the boolean sliding window approach of estimating probabilities.
# Since the `CoherenceModel` caches the accumulated statistics, calculation of c_uci and c_npmi are
# practically free after calculating c_v coherence. These two methods are simpler and were shown to
# correlate less with human judgements than c_v but more so than u_mass. <<< **this is not correct.
# c_uci and c_npmi use per default a different segmentation than c_v**
