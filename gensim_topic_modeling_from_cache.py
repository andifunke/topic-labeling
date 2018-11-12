import gc
from os import makedirs
from os.path import join, exists, dirname
import sys
import pandas as pd
import gensim
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel
from gensim.models.callbacks import CoherenceMetric, DiffMetric, PerplexityMetric, ConvergenceMetric
from constants import ETL_PATH, DATASETS
import logging
import json
import numpy as np
import argparse
from topic_reranking import NBTOPICS, PARAMS

# --- arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--logger", type=str, required=False, default='shell')
parser.add_argument("--params", nargs='*', type=str, required=False, default=PARAMS)
parser.add_argument("--nbtopics", nargs='*', type=int, required=False, default=NBTOPICS)
parser.add_argument("--nbfiles", type=int, required=False, default=None)
parser.add_argument("--epochs", type=int, required=False, default=20)
args = vars(parser.parse_args())

dataset = DATASETS[args['dataset']]
callback_logger = args['logger']
param_args = args['params']
nbtopics_args = args['nbtopics']
epochs = args['epochs']

# --- logging ---
LOG_PATH = './../logs/LDA_training_{}.log'.format(dataset)
makedirs(dirname(LOG_PATH), exist_ok=True)
logger = logging.getLogger('LDA_training')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# file logger
fh = logging.FileHandler(LOG_PATH)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# stdout logger
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info('python: ' + sys.version)
logger.info('gensim: ' + gensim.__version__)
logger.info('pandas: ' + pd.__version__)
logger.info(param_args)


def init_callbacks(training_corpus, test_corpus, documents, viz_env=None, title_suffix=''):
    pl_test = PerplexityMetric(
        corpus=test_corpus,
        logger=callback_logger, viz_env=viz_env,
        title="Perplexity (test)" + title_suffix
    )
    ch_umass = CoherenceMetric(
        corpus=training_corpus, coherence="u_mass", topn=10,
        logger=callback_logger, viz_env=viz_env,
        title="Coherence (u_mass)" + title_suffix
    )
    ch_cv = CoherenceMetric(
        corpus=training_corpus, texts=documents, coherence="c_v", topn=10,
        logger=callback_logger, viz_env=viz_env,
        title="Coherence (c_v)" + title_suffix
    )
    diff_kl = DiffMetric(
        distance="kullback_leibler",
        logger=callback_logger, viz_env=viz_env,
        title="Diff (kullback_leibler)" + title_suffix
    )
    convergence_kl = ConvergenceMetric(
        distance="jaccard",
        logger=callback_logger, viz_env=viz_env,
        title="Convergence (jaccard)" + title_suffix
    )
    return [pl_test, ch_umass, ch_cv, diff_kl, convergence_kl]


def get_parameterset(corpus, dictionary, callbacks=None, nbtopics=100, parametrization='a42',
                     eval_every=None):
    logger.info(
        'building LDA model "{}" with {} number of topics'
        .format(parametrization, nbtopics)
    )
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


def split_corpus(corpus):
    length = len(corpus)
    max_test_size_rel = 0.1
    max_test_size_abs = 5000
    if length * max_test_size_rel < max_test_size_abs:
        split_idx = int(length * (1 - max_test_size_rel))
    else:
        split_idx = length - max_test_size_abs
    logger.info(
        'split dataset. size of: train_set={:d}, test_set={:d}'.format(split_idx, length-split_idx)
    )
    return corpus[:split_idx], corpus[split_idx:]


def main():
    file_name = '{}_{}_nouns_{}'.format(dataset, 'fullset', 'bow')

    dict_path = join(ETL_PATH, 'LDAmodel', file_name + '.dict')
    dictionary = Dictionary.load(dict_path)

    doc_path = join(ETL_PATH, 'LDAmodel', file_name[:-3] + 'texts.json')
    with open(doc_path, 'r') as fp:
        texts = json.load(fp)

    corpus_path = join(ETL_PATH, 'LDAmodel', file_name + '.mm')
    corpus = MmCorpus(corpus_path)
    training_corpus, test_corpus = split_corpus(corpus)

    topn = 20
    model = 'LDAmodel'
    metrics = []
    for params in param_args:
        for nbtopics in nbtopics_args:
            gc.collect()

            callbacks = init_callbacks(
                documents=texts,
                training_corpus=training_corpus,
                test_corpus=test_corpus,
            )
            kwargs = get_parameterset(
                training_corpus,
                dictionary,
                callbacks=callbacks,
                nbtopics=nbtopics,
                parametrization=params
            )
            logger.info('running ' + model)
            ldamodel = LdaModel(**kwargs)

            model_dir = join(ETL_PATH, model, params)
            if not exists(model_dir):
                makedirs(model_dir)
            model_name = join(model_dir, '{}_{}_{}_{}'.format(dataset, model, params, nbtopics))

            # save topics
            topics = [
                [dataset] +
                [dictionary[term[0]] for term in ldamodel.get_topic_terms(i, topn=topn)]
                for i in range(nbtopics)
            ]
            df_lda = pd.DataFrame(topics, columns=['dataset'] + ['term' + str(i) for i in range(topn)])
            df_lda.to_csv(model_name + '.csv')

            # save metrics
            current_metrics = ldamodel.metrics
            metrics.append(('env_id', current_metrics))
            with open(model_name + '_metrics.json', 'w') as fp:
                serializable_metrics = {}
                for k, v in current_metrics.items():
                    if isinstance(v[0], np.ndarray):
                        serializable_metrics[k] = [x.tolist() for x in v]
                    else:
                        serializable_metrics[k] = [float(x) for x in v]
                json.dump(serializable_metrics, fp)

            # save model
            logger.info('saving to ' + model_name)
            ldamodel.callbacks = None
            ldamodel.save(model_name)

            gc.collect()


if __name__ == '__main__':
    main()
