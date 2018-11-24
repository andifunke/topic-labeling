import gc
from os.path import join
import pandas as pd
from gensim.models import CoherenceModel
from constants import DATASETS, TPX_PATH
from utils import load, init_logging


def isstr(x):
    return isinstance(x, str)


logger = init_logging(name='wiki_coherence', basic=False, to_stdout=False, to_file=True)
datasets = ['dewac1', 'O', 'news', 'speeches']
metrics = ['u_mass', 'c_v', 'c_npmi', 'c_uci']
wiki_dict = load('dict', 'dewiki', 'unfiltered', logger=logger)
wiki_texts = load('texts', 'dewiki', logger=logger)


for dataset in datasets:
    gc.collect()
    logger.info(dataset)

    topics = load(
        dataset,
        'topics',
        'e42',
        # 'ref',
        # 10,
        logger=logger
    )
    logger.info(f'number of topics: {len(topics)}')
    topics_values = topics.values

    in_dict = topics.applymap(lambda x: x in wiki_dict.token2id)
    oov = topics[~in_dict]
    oov = oov.apply(set)
    oov = set().union(*oov)
    oov = sorted(map(lambda x: [x], filter(isstr, oov)))
    if oov:
        wiki_dict.add_documents(oov)
        _ = wiki_dict[0]

    scores = dict()
    for metric in metrics:
        gc.collect()
        logger.info(metric)
        cm = CoherenceModel(
            topics=topics_values,
            dictionary=wiki_dict,
            texts=wiki_texts + oov,
            coherence=metric,
            topn=10,
            window_size=20,
            processes=4
        )
        coherence_scores = cm.get_coherence_per_topic(with_std=True, with_support=True)
        scores[metric + '_wiki'] = coherence_scores
        gc.collect()

    df = pd.DataFrame(scores)
    df.index = topics.index
    df.to_csv(join(TPX_PATH, f'{DATASETS.get(dataset, dataset)}_topic-wiki-scores.csv'))
    gc.collect()
