import gc
from itertools import chain
from os import makedirs
from os.path import join, exists

import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LsiModel

from constants import LDA_PATH, LSI_PATH
from train_lda import parse_args, split_corpus
from utils import init_logging, log_args


def main():
    # --- arguments ---
    (
        dataset, version, _, _, nbs_topics, _, _, cache_in_memory, use_callbacks, tfidf, args
    ) = parse_args()

    model_class = 'LSImodel'
    _split_ = "_split" if use_callbacks else ""

    data_name = f'{dataset}_{version}_{tfidf}'
    data_dir = join(LDA_PATH, version, tfidf)

    # --- logging ---
    logger = init_logging(name=data_name, basic=False, to_stdout=True, to_file=True)
    logg = logger.info
    log_args(logger, args)

    # --- load dict ---
    logg('Loading dictionary')
    data_file = join(data_dir, f'{data_name}.dict')
    dictionary = Dictionary.load(data_file)

    # --- load corpus ---
    logg('Loading corpus')
    data_file = join(data_dir, f'{data_name}.mm')
    corpus = MmCorpus(data_file)
    if cache_in_memory:
        logg('Reading corpus into RAM')
        corpus = list(corpus)
    if use_callbacks:
        train, test = split_corpus(corpus)
    else:
        train, test = corpus, []
    logg(f'size of... train_set={len(train)}, test_set={len(test)}')

    # --- train ---
    topn = 20
    columns = [f'term{x}' for x in range(topn)] + [f'weight{x}' for x in range(topn)]
    for nbtopics in nbs_topics:
        gc.collect()

        logg(f'Running {model_class} with {nbtopics} topics')
        model = LsiModel(corpus=train, num_topics=nbtopics, id2word=dictionary)

        model_dir = join(LSI_PATH, version, tfidf, f'{_split_}')
        model_path = join(model_dir, f'{dataset}_{model_class}{_split_}_{nbtopics}')
        if not exists(model_dir):
            makedirs(model_dir)

        # --- save topics ---
        topics = model.show_topics(num_words=topn, formatted=False)
        topics = [list(chain(*zip(*topic[1]))) for topic in topics]
        topics = pd.DataFrame(topics, columns=columns)
        logg(f'Saving topics to {model_path}.csv')
        topics.to_csv(f'{model_path}.csv')

        # --- save model ---
        logg(f'Saving model to {model_path}')
        model.save(model_path)

    # --- done ---
    logg(
        f'\n'
        f'----- end -----\n'
        f'----- {dataset.upper()} -----\n'
        f'{"#" * 50}\n'
    )


if __name__ == '__main__':
    main()
