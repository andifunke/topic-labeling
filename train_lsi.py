import gc
from os import makedirs
from os.path import join, exists

import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LsiModel

from constants import LDA_PATH, ETL_PATH
from train_lda import parse_args, split_corpus
from utils import init_logging, log_args


def main():
    # --- arguments ---
    (
        dataset, version, _, _, nbs_topics, _, _, cache_in_memory, use_callbacks, tfidf, args
    ) = parse_args()

    model_class = 'LSImodel'
    _tfidf_ = "tfidf" if tfidf else "bow"
    _split_ = "_split" if use_callbacks else ""

    data_name = f'{dataset}_{version}_{_tfidf_}'
    data_dir = join(LDA_PATH, version, _tfidf_)

    # --- logging ---
    logger = init_logging(name=data_name, basic=False, to_stdout=True, to_file=True)
    log = logger.info
    log_args(logger, args)

    # --- load dict ---
    log('Loading dictionary')
    data_file = join(data_dir, f'{data_name}.dict')
    dictionary = Dictionary.load(data_file)

    # --- load corpus ---
    log('Loading corpus')
    data_file = join(data_dir, f'{data_name}.mm')
    corpus = MmCorpus(data_file)
    if cache_in_memory:
        log('Reading corpus into RAM')
        corpus = list(corpus)
    if use_callbacks:
        train, test = split_corpus(corpus)
    else:
        train, test = corpus, []
    log(f'size of... train_set={len(train)}, test_set={len(test)}')

    # --- train ---
    for nbtopics in nbs_topics:
        gc.collect()

        log(f'Running {model_class} with {nbtopics} topics')
        model = LsiModel(corpus=train, num_topics=nbtopics, id2word=dictionary)

        model_dir = join(ETL_PATH, model_class, version, _tfidf_, f'{_split_}')
        model_name = join(model_dir, f'{dataset}_{model_class}{_split_}_{nbtopics}')
        if not exists(model_dir):
            makedirs(model_dir)

        # --- save topics ---
        topics = model.show_topics(formatted=False)
        topics = (
            pd
            .DataFrame([t[1] for t in topics])
            .stack()
            .apply(pd.Series)
            .rename(columns={0: 'terms', 1: 'weights'})
            .unstack()
        )
        log(f'Saving topics to {model_name}.csv')
        topics.to_csv(f'{model_name}.csv')

        # --- save model ---
        log(f'Saving model to {model_name}')
        model.save(model_name)

    # --- done ---
    log(
        f'\n'
        f'----- end -----\n'
        f'----- {dataset.upper()} -----\n'
        f'{"#" * 50}\n'
    )


if __name__ == '__main__':
    main()
