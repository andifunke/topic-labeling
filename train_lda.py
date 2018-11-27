import gc
from os import makedirs
from os.path import join, exists
import json
import argparse

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.matutils import kullback_leibler, hellinger, jaccard_distance, jensen_shannon
from gensim.models import LdaModel, CoherenceModel
from gensim.models.callbacks import Metric

from constants import DATASETS, NBTOPICS, PARAMS, LDA_PATH
from utils import init_logging, log_args

np.set_printoptions(precision=3, threshold=11, formatter={'float': '{: 0.3f}'.format})
LOG = None


class EpochLogger(Metric):
    """Callback to log information about training"""
    def __init__(self, title=None, message='', log=None):
        self.logger = None
        self.viz_env = None
        self.title = title
        self.message = message
        self.epoch = 1
        self.log = log

    def get_value(self, **kwargs):
        if self.log is not None:
            self.log(f"--- {self.title} --- Epoch #{self.epoch:02d} --- {self.message} ---")
        self.epoch += 1
        gc.collect()


class PerplexityMetric(Metric):
    """Metric class for perplexity evaluation."""
    def __init__(self, corpus=None, logger=None, viz_env=None, title=None, log=None):
        self.corpus = corpus
        self.logger = logger
        self.viz_env = viz_env
        self.title = title
        self.log = log

    def get_value(self, **kwargs):
        super(PerplexityMetric, self).set_parameters(**kwargs)
        if self.log is not None:
            self.log('  %s' % self.title)
        corpus_words = sum(cnt for document in self.corpus for _, cnt in document)
        perwordbound = self.model.bound(self.corpus) / corpus_words
        value = np.exp2(-perwordbound)
        if self.log is not None:
            self.log('    %r' % value)
        return value


class CoherenceMetric(Metric):
    """Metric class for coherence evaluation."""
    def __init__(
            self, corpus=None, texts=None, dictionary=None, coherence=None,
            window_size=None, topn=10, logger=None, viz_env=None, title=None,
            log=None, processes=-1
    ):
        self.corpus = corpus
        self.dictionary = dictionary
        self.coherence = coherence
        self.texts = texts
        self.window_size = window_size
        self.topn = topn
        self.logger = logger
        self.viz_env = viz_env
        self.title = title
        self.model = None
        self.topics = None
        self.processes = processes
        self.log = log

    def get_value(self, **kwargs):
        super(CoherenceMetric, self).set_parameters(**kwargs)
        if self.log is not None:
            self.log('  %s' % self.title)
        cm = CoherenceModel(
            model=self.model, topics=self.topics, texts=self.texts, corpus=self.corpus,
            dictionary=self.dictionary, window_size=self.window_size,
            coherence=self.coherence, topn=self.topn, processes=self.processes
        )
        value = cm.get_coherence()
        if self.log is not None:
            self.log('    %r' % value)
        return value


class DiffMetric(Metric):
    """Metric class for topic difference evaluation."""
    def __init__(
            self, distance="jaccard", num_words=100, n_ann_terms=10, diagonal=True,
            annotation=False, normed=True, logger=None, viz_env=None, title=None,
            convergence=False, log=None
    ):
        self.distance = distance
        self.num_words = num_words
        self.n_ann_terms = n_ann_terms
        self.diagonal = diagonal
        self.annotation = annotation
        self.normed = normed
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

        self.convergence = convergence
        self.log = log

        self.prev_get_topics = None
        self.prev_topics = None

    def get_value(self, **kwargs):
        super(DiffMetric, self).set_parameters(**kwargs)
        if self.log is not None:
            self.log('  %s' % self.title)
        diff_diagonal, _ = self.diff(
            self.distance, self.num_words, self.n_ann_terms,
            self.diagonal, self.annotation, self.normed
        )
        if self.convergence:
            value = np.sum(diff_diagonal)
        else:
            value = diff_diagonal
        if self.log is not None:
            if isinstance(value, np.ndarray):
                self.log('    %r' % value[:5])
            else:
                self.log('    %s' % value)

        self.set_prev_topics()
        return value

    def set_prev_topics(self):
        self.prev_get_topics = self.model.get_topics()
        t2_size = self.prev_get_topics.shape[0]
        self.prev_topics = [
            {w for (w, _) in self.model.show_topic(topic, topn=self.num_words)}
            for topic in range(t2_size)
        ]

    def diff(self, distance="kullback_leibler", num_words=100,
             n_ann_terms=10, diagonal=False, annotation=True, normed=True):
        """Calculate the difference in topic distributions between two models: `self` and `other`."""
        distances = {
            "kullback_leibler": kullback_leibler,
            "hellinger": hellinger,
            "jaccard": jaccard_distance,
            "jensen_shannon": jensen_shannon
        }

        if distance not in distances:
            valid_keys = ", ".join("`{}`".format(x) for x in distances.keys())
            raise ValueError("Incorrect distance, valid only {}".format(valid_keys))

        distance_func = distances[distance]
        d1, d2 = self.model.get_topics(), self.prev_get_topics
        if d2 is None:
            d2 = np.random.normal(1, 1, d1.shape) / d1.shape[1]
        t1_size, t2_size = d1.shape[0], d1.shape[0]
        annotation_terms = None

        fst_topics = [
            {w for (w, _) in self.model.show_topic(topic, topn=num_words)} for topic in range(t1_size)
        ]
        snd_topics = self.prev_topics
        if snd_topics is None:
            snd_topics = [set() for i in fst_topics]

        if distance == "jaccard":
            d1, d2 = fst_topics, snd_topics

        if diagonal:
            assert t1_size == t2_size, \
                "Both input models should have same no. of topics, " \
                "as the diagonal will only be valid in a square matrix"
            # initialize z and annotation array
            z = np.zeros(t1_size)
            if annotation:
                annotation_terms = np.zeros(t1_size, dtype=list)
        else:
            # initialize z and annotation matrix
            z = np.zeros((t1_size, t2_size))
            if annotation:
                annotation_terms = np.zeros((t1_size, t2_size), dtype=list)

        # iterate over each cell in the initialized z and annotation
        for topic in np.ndindex(z.shape):
            topic1 = topic[0]
            if diagonal:
                topic2 = topic1
            else:
                topic2 = topic[1]

            z[topic] = distance_func(d1[topic1], d2[topic2])
            if annotation:
                pos_tokens = fst_topics[topic1] & snd_topics[topic2]
                neg_tokens = fst_topics[topic1].symmetric_difference(snd_topics[topic2])

                pos_tokens = list(pos_tokens)[:min(len(pos_tokens), n_ann_terms)]
                neg_tokens = list(neg_tokens)[:min(len(neg_tokens), n_ann_terms)]

                annotation_terms[topic] = [pos_tokens, neg_tokens]

        if normed:
            if np.max(z) == np.inf:
                z = np.ones_like(z)
            elif np.abs(np.max(z)) > 1e-8:
                z /= np.max(z)

        return z, annotation_terms


def init_callbacks(
        dataset, callback_logger, training_corpus, test_corpus, documents,
        viz_env=None, title_suffix='', processes=-1,
        version=None, param=None, nbtopics=None, tfidf='',
):
    return [
        EpochLogger(
            title=dataset,
            message=f'[{version}, {param}, {nbtopics}, {tfidf}]'
                    f' calculating metrics',
            log=LOG
        ),
        PerplexityMetric(
            corpus=test_corpus,
            logger=callback_logger, viz_env=viz_env,
            title="Perplexity (test)" + title_suffix,
            log=LOG
        ),
        CoherenceMetric(
            corpus=training_corpus, coherence="u_mass", topn=10,
            logger=callback_logger, viz_env=viz_env,
            title="Coherence (u_mass)" + title_suffix,
            log=LOG, processes=processes
        ),
        CoherenceMetric(
            corpus=training_corpus, texts=documents, coherence="c_v", topn=10,
            logger=callback_logger, viz_env=viz_env,
            title="Coherence (c_v)" + title_suffix,
            log=LOG, processes=processes
        ),
        DiffMetric(
            distance="kullback_leibler",
            logger=callback_logger, viz_env=viz_env,
            title="Diff (kullback_leibler)" + title_suffix,
            log=LOG
        ),
        DiffMetric(
            distance="jaccard",
            logger=callback_logger, viz_env=viz_env,
            title="Convergence (jaccard)" + title_suffix,
            convergence=True,
            log=LOG
        ),
        EpochLogger(
            title=dataset,
            message=f'[{version}, {param}, {nbtopics}, {tfidf}]'
                    f' epoch finished',
            log=LOG
        ),
    ]


def get_parameterset(
        corpus, dictionary, callbacks=None, nbtopics=100, parametrization='a42',
        eval_every=None, epochs=20
):
    default = dict(
        random_state=42, corpus=corpus, id2word=dictionary, num_topics=nbtopics,
        eval_every=eval_every, callbacks=callbacks, chunksize=20000, dtype=np.float64
    )
    ldamodels = {
        'a42': dict(passes=epochs),
        'b42': dict(passes=epochs, iterations=200),
        'c42': dict(passes=epochs, iterations=1000),
        'd42': dict(passes=epochs, iterations=200, alpha=0.1, eta=0.01),
        'e42': dict(passes=epochs, iterations=200, alpha='auto', eta='auto'),
    }
    for key, dic in ldamodels.items():
        dic.update(default)
    return ldamodels[parametrization]


def split_corpus(corpus, max_test_size_rel=0.1, max_test_size_abs=5000):
    length = len(corpus)
    if length * max_test_size_rel < max_test_size_abs:
        split_idx = int(length * (1 - max_test_size_rel))
    else:
        split_idx = length - max_test_size_abs
    train, test = corpus[:split_idx], corpus[split_idx:]
    return train, test


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--version", type=str, required=False, default='default')
    parser.add_argument("--logger", type=str, required=False, default='shell')
    parser.add_argument("--params", nargs='*', type=str, required=False, default=PARAMS)
    parser.add_argument("--nbtopics", nargs='*', type=int, required=False, default=NBTOPICS)
    parser.add_argument("--epochs", type=int, required=False, default=20)
    parser.add_argument("--cores", type=int, required=False, default=4)

    parser.add_argument('--cacheinmem', dest='cache_in_memory', action='store_true', required=False)
    parser.add_argument('--no-cacheinmem', dest='cache_in_memory', action='store_false', required=False)
    parser.set_defaults(cache_in_memory=False)
    parser.add_argument('--callbacks', dest='use_callbacks', action='store_true', required=False)
    parser.add_argument('--no-callbacks', dest='use_callbacks', action='store_false', required=False)
    parser.set_defaults(use_callbacks=True)
    parser.add_argument('--tfidf', dest='tfidf', action='store_true', required=False)
    parser.add_argument('--no-tfidf', dest='tfidf', action='store_false', required=False)
    parser.set_defaults(tfidf=False)

    args = parser.parse_args()

    args.dataset = DATASETS.get(args.dataset, args.dataset)

    return (
        args.dataset, args.version, args.logger, args.params, args.nbtopics, args.epochs,
        args.cores, args.cache_in_memory, args.use_callbacks, args.tfidf, args
    )


def main():
    global LOG

    # --- arguments ---
    (
        dataset, version, cb_logger, params, nbs_topics, epochs,
        cores, cache_in_memory, use_callbacks, tfidf, args
    ) = parse_args()

    model_class = 'LDAmodel'
    _tfidf_ = "tfidf" if tfidf else "bow"
    _split_ = "_split" if use_callbacks else ""

    data_name = f'{dataset}_{version}'
    data_dir = join(LDA_PATH, version)

    # --- logging ---
    logger = init_logging(
        name=f'LDA_{data_name}_{_tfidf_}{_split_}_ep{epochs}',
        basic=False, to_stdout=True, to_file=True
    )
    LOG = logger.info
    log_args(logger, args)

    # --- load texts ---
    if use_callbacks:
        LOG('Loading texts')
        file_path = join(data_dir, f'{data_name}_texts.json')
        with open(file_path, 'r') as fp:
            texts = json.load(fp)
    else:
        texts = []

    data_dir = join(data_dir, _tfidf_)
    data_name = f'{data_name}_{_tfidf_}'

    # --- load dict ---
    LOG('Loading dictionary')
    file_path = join(data_dir, f'{data_name}.dict')
    dictionary = Dictionary.load(file_path)

    # --- load corpus ---
    LOG('Loading corpus')
    file_path = join(data_dir, f'{data_name}.mm')
    corpus = MmCorpus(file_path)
    if cache_in_memory:
        LOG('Reading corpus into RAM')
        corpus = list(corpus)
    if use_callbacks:
        train, test = split_corpus(corpus)
    else:
        train, test = corpus, []
    LOG(f'size of... train_set={len(train)}, test_set={len(test)}')

    # --- enable visdom ---
    vis = None
    if cb_logger == 'visdom':
        import visdom
        vis = visdom.Visdom()

    # --- train ---
    topn = 20
    metrics = []
    for param in params:
        env_id = f"{dataset}-{model_class}"
        for nbtopics in nbs_topics:
            gc.collect()

            LOG('Initializing Callbacks')
            callbacks = init_callbacks(
                dataset=dataset,
                callback_logger=cb_logger,
                documents=texts,
                training_corpus=train,
                test_corpus=test,
                processes=cores,
                version=version,
                param=param,
                nbtopics=nbtopics,
                tfidf=_tfidf_
            )
            if not use_callbacks:
                callbacks = callbacks[-1:]

            kwargs = get_parameterset(
                train,
                dictionary,
                callbacks=callbacks,
                nbtopics=nbtopics,
                parametrization=param,
                epochs=epochs
            )

            LOG(f'Running {model_class} {_tfidf_} "{param}{_split_}" with {nbtopics} topics')
            ldamodel = LdaModel(**kwargs)
            gc.collect()

            model_dir = join(data_dir, f'{param}{_split_}')
            model_name = join(model_dir, f'{dataset}_LDAmodel_{param}{_split_}_{nbtopics}_ep{epochs}')
            if not exists(model_dir):
                makedirs(model_dir)

            # --- save topics ---
            topics = [
                [dataset] +
                [dictionary[term[0]] for term in ldamodel.get_topic_terms(i, topn=topn)]
                for i in range(nbtopics)
            ]
            df_lda = pd.DataFrame(topics, columns=['dataset'] + ['term' + str(i) for i in range(topn)])
            LOG(f'Saving topics to {model_name}.csv')
            df_lda.to_csv(f'{model_name}.csv')

            # --- save metrics ---
            current_metrics = ldamodel.metrics
            metrics.append(('env_id', current_metrics))
            with open(f'{model_name}_metrics.json', 'w') as fp:
                serializable_metrics = {}
                for k, v in current_metrics.items():
                    if k == dataset:
                        continue
                    if isinstance(v[0], np.ndarray):
                        serializable_metrics[k] = [x.tolist() for x in v]
                    else:
                        serializable_metrics[k] = [float(x) for x in v]
                LOG(f'Saving metrics to {model_name}_metrics.json')
                json.dump(serializable_metrics, fp)

            # --- save model ---
            LOG(f'Saving LDAmodel to {model_name}')
            ldamodel.callbacks = None
            ldamodel.save(model_name)

            # --- save visdom environment ---
            if vis is not None:
                vis.save([env_id])

            gc.collect()

    # --- done ---
    LOG(
        f'\n'
        f'----- end -----\n'
        f'----- {dataset.upper()} -----\n'
        f'{"#" * 50}\n'
    )


if __name__ == '__main__':
    main()
