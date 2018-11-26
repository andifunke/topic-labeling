import json
import logging
import re
import sys
from genericpath import exists
from os import makedirs
from os.path import join
from pprint import pformat

import gensim
import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import Doc2Vec, Word2Vec, FastText

from constants import (
    ETL_PATH, NLP_PATH, SMPL_PATH, LDA_PATH, DSETS, PARAMS, NBTOPICS, METRICS, VERSIONS,
    EMB_PATH
)

try:
    from tabulate import tabulate
except ImportError as ie:
    print(ie)


def tprint(df, head=0, floatfmt=None, to_latex=False):
    if df is None:
        return
    shape = df.shape
    if head > 0:
        df = df.head(head)
    elif head < 0:
        df = df.tail(-head)
    kwargs = dict()
    if floatfmt is not None:
        kwargs['floatfmt'] = floatfmt
    try:
        print(tabulate(df, headers="keys", tablefmt="pipe", showindex="always", **kwargs))
    except:
        print(df)
    print('shape:', shape, '\n')

    if to_latex:
        print(df.to_latex(bold_rows=True))


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


def init_logging(name='', basic=True, to_stdout=False, to_file=True, log_file=None, log_dir='../logs'):

    if log_file is None:
        log_file = name+'.log' if name else 'train.log'

    if basic:
        if to_file:
            if not exists(log_dir):
                makedirs(log_dir)
            file_path = join(log_dir, log_file)
            logging.basicConfig(
                filename=file_path,
                format='%(asctime)s - %(name)s - %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                level=logging.INFO
            )
        else:
            logging.basicConfig(
                format='%(asctime)s - %(name)s - %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                level=logging.INFO
            )
        logger = logging.getLogger()

    else:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        if to_file:
            # create path if necessary
            if not exists(log_dir):
                makedirs(log_dir)
            file_path = join(log_dir, log_file)
            fh = logging.FileHandler(file_path)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        if to_stdout:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    logger.info('')
    logger.info('#' * 50)
    logger.info('----- %s -----' % name.upper())
    logger.info('----- start -----')
    logger.info('python: ' + sys.version.replace('\n', ' '))
    logger.info('pandas: ' + pd.__version__)
    logger.info('gensim: ' + gensim.__version__)

    return logger


def log_args(logger, args):
    logger.info('\n' + pformat(vars(args)))


def load(*args, logger=None):
    """
    work in progress: may not work for all cases, especially not yet for reading distributed
    datsets like dewiki and dewac.
    """

    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    if not args:
        log('no arguments, no load')
        return

    single = {
        'hashmap': join(ETL_PATH, 'dewiki_hashmap.pickle'),
        'meta': join(ETL_PATH, 'dewiki_metadata.pickle'),
        'phrases': join(ETL_PATH, 'dewiki_phrases_lemmatized.pickle'),
        'links': join(ETL_PATH, 'dewiki_links.pickle'),
        'categories': join(ETL_PATH, 'dewiki_categories.pickle'),
        'disamb': join(ETL_PATH, 'dewiki_disambiguation.pickle'),
    }
    dataset = None
    purposes = {
        'goodids', 'etl', 'nlp', 'simple', 'smpl', 'wiki_phrases', 'embedding',
        'topic', 'topics', 'label', 'labels', 'lda', 'ldamodel', 'score', 'scores',
        'lemmap', 'disamb', 'dict', 'corpus', 'texts', 'wiki_scores', 'x2v_scores'
    }
    purpose = None
    version = None
    params = []
    nbtopics = []
    metrics = []
    file = None

    if isinstance(args, str):
        args = [args]
    args = [arg.replace('-', '_') if isinstance(arg, str) else arg for arg in args]
    # log(args)

    # parse args
    for arg in args:
        if arg in single:
            purpose = 'single'
            file = single[arg]
            dataset = True
            break
        elif not dataset and arg.lower() in DSETS:
            dataset = DSETS[arg]
        elif not dataset and arg in DSETS.values():
            dataset = arg
        elif not purpose and arg.lower() in purposes:
            purpose = arg.lower()
        elif any([s in arg for s in ['d2v', 'w2v', 'ftx'] if isinstance(arg, str)]):
            purpose = 'embedding'
            dataset = arg
        elif arg in PARAMS:
            params.append(arg)
        elif arg in NBTOPICS:
            nbtopics.append(arg)
        elif arg in METRICS:
            metrics.append(arg)
        elif not version and arg in VERSIONS:
            version = arg

    # setting default values
    if not version:
        version = 'noun'
    if 'default' in args:
        params.append('e42')
        nbtopics.append('100')
        metrics.append('ref')

    log('purpose ' + purpose)
    log('dataset ' + dataset)

    # combine args
    if purpose == 'single':
        pass
    elif purpose == 'goodids' and dataset in ['dewac', 'dewiki']:
        file = join(ETL_PATH, f'{dataset}_good_ids.pickle')
    elif purpose == 'lemmap':
        file = join(ETL_PATH, f'{dataset}_lemmatization_map.pickle')
    elif purpose == 'embedding':
        file = join(EMB_PATH, dataset, dataset)
    elif purpose in {'topic', 'topics'}:
        file = join(LDA_PATH, version, 'topics', f'{dataset}_topic-candidates.csv')
    elif purpose in {'score', 'scores'}:
        file = join(LDA_PATH, version, 'topics', f'{dataset}_topic-scores.csv')
    elif purpose == 'wiki_scores':
        file = join(LDA_PATH, version, 'topics', f'{dataset}_topic-wiki-scores.csv')
    elif purpose == 'x2v_scores':
        file = join(LDA_PATH, version, 'topics', f'{dataset}_topic-x2v-scores.csv')
    elif purpose in {'label', 'labels'}:
        file = join(LDA_PATH, version, 'topics', f'{dataset}_label-candidates_full.csv')
    elif purpose == 'dict':
        if dataset == 'dewiki' and 'unfiltered' in args:
            dict_path = join(LDA_PATH, version, f'dewiki_noun_bow_unfiltered.dict')
        else:
            dict_path = join(LDA_PATH, version, f'{dataset}_{version}_bow.dict')
        log(f'loading dict from {dict_path}')
        dict_from_corpus = Dictionary.load(dict_path)
        _ = dict_from_corpus[0]  # init dictionary
        return dict_from_corpus
    elif purpose == 'corpus':
        corpus_path = join(LDA_PATH, version, f'{dataset}_{version}_bow.mm')
        log(f'loading corpus from {corpus_path}')
        corpus = MmCorpus(corpus_path)
        corpus = list(corpus)
        return corpus
    elif purpose == 'texts':
        doc_path = join(LDA_PATH, version, f'{dataset}_{version}_bow_texts.json')
        with open(doc_path, 'r') as fp:
            log(f'loading texts from {doc_path}')
            texts = json.load(fp)
        return texts
    elif purpose in {'nlp', 'simple', 'smpl', 'wiki', 'wiki_phrases', 'phrases', 'etl', None}:
        if purpose in {'etl', None}:
            directory = ETL_PATH
            suffix = ''
        elif purpose == 'nlp':
            directory = NLP_PATH
            suffix = '_nlp'
        elif purpose in {'simple', 'smpl'}:
            directory = SMPL_PATH
            suffix = '_simple'
        elif purpose in {'wiki', 'wiki_phrases', 'phrases'}:
            directory = join(SMPL_PATH, 'wiki_phrases')
            suffix = '_simple_wiki_phrases'
        else:
            log('oops')
            return
        if dataset.lower() in {'speeches', 's'}:
            file = [
                join(directory, f'{DSETS["E"]}{suffix}.pickle'),
                join(directory, f'{DSETS["P"]}{suffix}.pickle')
            ]
        elif dataset.lower() in {'news', 'n', 'f'}:
            file = [
                join(directory, f'{DSETS["FA"]}{suffix}.pickle'),
                join(directory, f'{DSETS["FO"]}{suffix}.pickle')
            ]
        else:
            # TODO: allow to load full or partially dewiki and dewac
            file = join(directory, f'{dataset.replace("dewac1", "dewac_01")}{suffix}.pickle')

    try:
        log(f'Reading {file}')
        if purpose == 'embedding':
            if 'd2v' in dataset:
                return Doc2Vec.load(file)
            if 'w2v' in dataset:
                return Word2Vec.load(file)
            if 'ftx' in dataset:
                return FastText.load(file)
        elif isinstance(file, str) and file.endswith('.pickle'):
            df = pd.read_pickle(file)
            if purpose == 'single' and 'phrases' in args and 'minimal' in args:
                pat = re.compile(r'^[a-zA-ZÄÖÜäöü]+.*')
                df = df.set_index('token').text
                df = df[df.str.match(pat)]
            return df
        elif isinstance(file, str) and file.endswith('.csv'):
            index = None
            header = 0
            if purpose in {'label', 'labels'}:
                index = [0, 1, 2, 3, 4, 5]
            elif purpose in {'topic', 'topics', 'score', 'scores', 'x2v_scores'}:
                index = [0, 1, 2, 3, 4]
            elif purpose in {'wiki_scores'}:
                index = [0, 1, 2, 3, 4]
                header = 1

            df = pd.read_csv(file, index_col=index, header=header)

            if len(metrics) > 0:
                df = df.query('metric in @metrics')
            if len(params) > 0:
                df = df.query('param_id in @params')
            if len(nbtopics) > 0:
                df = df.query('nb_topics in @nbtopics')
            if purpose in {'label', 'labels'}:
                df = df.applymap(eval)
                if 'minimal' in args:
                    df = (
                        df.query('label_method == "comb"')
                        .reset_index(drop=True)
                        .applymap(lambda x: x[0])
                    )
            if purpose in {'topic', 'topics', 'score', 'scores', 'wiki_scores', 'x2v_scores'}:
                if 'minimal' in args:
                    df = df.reset_index(drop=True)
            return df
        else:
            df = pd.concat([pd.read_pickle(f) for f in file])
            return df
    except Exception as e:
        log(e)


def main():
    data = load('dewa1', 'topics')
    print(data)


if __name__ == '__main__':
    main()
