import re

import pandas as pd
from gensim.models import Doc2Vec, Word2Vec, FastText

from constants import ETL_PATH, NLP_PATH, SMPL_PATH, DATASETS_X, PARAMS, NBTOPICS, METRICS, VERSIONS
from os.path import join

try:
    from tabulate import tabulate
except ImportError:
    pass
    

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
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def load(*args):
    """
    work in progress: may not work for all cases, especially not yet for reading distributed
    datsets like dewiki and dewac.
    """
    if not args:
        print('no arguments, no load')
        return

    single = {
        'hashmap': join(ETL_PATH, 'dewiki_hashmap.pickle'),
        'meta': join(ETL_PATH, 'dewiki_metadata.pickle'),
        'phrases': join(ETL_PATH, 'dewiki_phrases_lemmatized.pickle'),
        'links': join(ETL_PATH, 'dewiki_links.pickle'),
        'categories': join(ETL_PATH, 'dewiki_categories.pickle'),
    }
    dataset = None
    purposes = {
        'goodids', 'etl', 'nlp', 'simple', 'smpl', 'wiki_phrases', 'embedding',
        'topic', 'topics', 'label', 'labels', 'lda', 'ldamodel', 'score', 'scores'
    }
    purpose = None
    version = None
    params = []
    nbtopics = []
    metrics = []
    file = None

    if isinstance(args, str):
        args = [args]
    # args = [x.lower() if isinstance(x, str) else x for x in args]
    # print(args)

    # parse args
    for arg in args:
        if arg in single:
            purpose = 'single'
            file = single[arg]
            dataset = True
            break
        elif not dataset and arg in DATASETS_X:
            dataset = DATASETS_X[arg]
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
    # if not params:
    #     params.append('e42')
    # if not nbtopics:
    #     nbtopics.append(100)
    # if not metrics:
    #     metrics.append('ref')

    # print(purpose)
    # print(dataset)

    # combine args
    if purpose == 'single':
        pass
    elif purpose == 'goodids' and dataset in ['dewac', 'dewiki']:
        file = join(ETL_PATH, f'{dataset}_good_ids.pickle')
    elif purpose == 'embedding':
        file = join(ETL_PATH, 'embeddings', dataset, dataset)
    elif purpose == 'nlp':
        file = join(NLP_PATH, f'{dataset}_nlp.pickle')
    elif purpose in {'simple', 'smpl'}:
        file = join(SMPL_PATH, f'{dataset}_simple.pickle')
    elif purpose == 'wiki_phrases':
        file = join(SMPL_PATH, 'wiki_phrases', f'{dataset}_simple_wiki_phrases.pickle')
    elif purpose in {'topic', 'topics'}:
        file = join(ETL_PATH, 'LDAmodel', version, 'Reranker', f'{dataset}_topic-candidates.csv')
    elif purpose in {'score', 'scores'}:
        file = join(ETL_PATH, 'LDAmodel', version, 'Reranker', f'{dataset}_evaluation-scores.csv')
    elif purpose in {'label', 'labels'}:
        file = join(ETL_PATH, 'LDAmodel', version, 'Reranker', f'{dataset}_label-candidates_full.csv')
    else:
        file = join(ETL_PATH, f'{dataset}.pickle')

    try:
        print('Reading', file)
        if purpose == 'embedding':
            if 'd2v' in dataset:
                return Doc2Vec.load(file)
            if 'w2v' in dataset:
                return Word2Vec.load(file)
            if 'ftx' in dataset:
                return FastText.load(file)
        elif file.endswith('.pickle'):
            df = pd.read_pickle(file)
            if purpose == 'single' and 'phrases' in args and 'minimal' in args:
                pat = re.compile(r'^[a-zA-ZÄÖÜäöü]+.*')
                df = df.set_index('token').text
                df = df[df.str.match(pat)]
            return df
        elif file.endswith('.csv'):
            index = None
            if purpose in {'label', 'labels'}:
                index = [0, 1, 2, 3, 4, 5]
            elif purpose in {'topic', 'topics', 'score', 'scores'}:
                index = [0, 1, 2, 3, 4]
            df = pd.read_csv(file, index_col=index)
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
            if purpose in {'topic', 'topics', 'score', 'scores'}:
                if 'minimal' in args:
                    df = df.reset_index(drop=True)
            return df
    except Exception as e:
        print(e)


def main():
    print(load('labels', 'O', 100, 'e42', 'ref', 'small').head())


if __name__ == '__main__':
    main()
