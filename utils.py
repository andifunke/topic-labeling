import json
import logging
import re
import sys
from genericpath import exists
from os import makedirs, listdir
from os.path import join
from pprint import pformat

import gensim
import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import Doc2Vec, Word2Vec, FastText, LdaModel

from constants import (
    ETL_PATH, NLP_PATH, SMPL_PATH, LDA_PATH, DSETS, PARAMS, NBTOPICS, METRICS, VERSIONS,
    EMB_PATH, CORPUS_TYPE, NOUN_PATTERN, BAD_TOKENS, PLACEHOLDER)

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


def multiload(dataset, purpose='etl'):
    if dataset.lower().startswith('dewa'):
        dewac = True
    elif dataset.lower().startswith('dewi'):
        dewac = False
    else:
        print('unkown dataset')
        return

    if purpose.lower() in ['simple', 'smpl', 'phrase']:
        if dewac:
            dpath = join(SMPL_PATH, 'wiki_phrases')
            pattern = re.compile(r'^dewac_[0-9]{2}_simple_wiki_phrases\.pickle')
            files = sorted([join(dpath, f) for f in listdir(dpath) if pattern.match(f)])
        else:
            dpath = join(SMPL_PATH, 'dewiki')
            pattern = re.compile(r'^dewiki_[0-9]+_[0-9]+__[0-9]+_simple\.pickle')
            files = sorted([join(dpath, f) for f in listdir(dpath) if pattern.match(f)])
    elif purpose.lower() == 'nlp':
        dpath = NLP_PATH
        if dewac:
            pattern = re.compile(r'^dewac_[0-9]{2}_nlp\.pickle')
            files = sorted([join(dpath, f) for f in listdir(dpath) if pattern.match(f)])
        else:
            pattern = re.compile(r'^dewiki_[0-9]+_[0-9]+_nlp\.pickle')
            files = sorted([join(dpath, f) for f in listdir(dpath) if pattern.match(f)])
    else:
        dpath = ETL_PATH
        if dewac:
            pattern = re.compile(r'^dewac_[0-9]{2}\.pickle')
            files = sorted([join(dpath, f) for f in listdir(dpath) if pattern.match(f)])
        else:
            files = [join(dpath, 'dewiki.pickle')]

    length = len(files)
    for i, file in enumerate(files, 1):
        print(f'Reading {i:02d}/{length}: {file}')
        yield pd.read_pickle(file)


def load(*args, logger=None):
    """
    work in progress: may not work for all cases, especially not yet for reading distributed
    datsets like dewiki and dewac.
    """

    def logg(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    if not args:
        logg('no arguments, no load')
        return

    single = {
        'hashmap': join(ETL_PATH, 'dewiki_hashmap.pickle'),
        'meta': join(ETL_PATH, 'dewiki_metadata.pickle'),
        'phrases': join(ETL_PATH, 'dewiki_phrases_lemmatized.pickle'),
        'links': join(ETL_PATH, 'dewiki_links.pickle'),
        'categories': join(ETL_PATH, 'dewiki_categories.pickle'),
        'disamb': join(ETL_PATH, 'dewiki_disambiguation.pickle'),
        'wikt': join(ETL_PATH, 'wiktionary_lemmatization_map.pickle'),
    }
    dataset = None
    purposes = {
        'goodids', 'etl', 'nlp', 'simple', 'smpl', 'wiki_phrases', 'embedding',
        'topic', 'topics', 'label', 'labels', 'lda', 'ldamodel', 'score', 'scores',
        'lemmap', 'disamb', 'dict', 'corpus', 'texts', 'wiki_scores', 'x2v_scores'
    }
    purpose = None
    version = None
    corpus_type = None
    params = []
    nbtopics = []
    metrics = []
    file = None

    if isinstance(args, str):
        args = [args]
    args = [arg.replace('-', '_') if isinstance(arg, str) else arg for arg in args]
    # logg(args)

    # parse args
    for arg in args:
        if arg in single:
            if arg == 'phrases' and 'lemmap' in args:
                dataset = 'dewiki_phrases'
                purpose = 'lemmap'
            else:
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
        elif not corpus_type and arg in CORPUS_TYPE:
            corpus_type = arg

    # setting default values
    if version is None:
        version = 'noun'
    if corpus_type is None:
        corpus_type = 'bow'
    if 'default' in args:
        params.append('e42')
        nbtopics.append('100')
        metrics.append('ref')

    # logg(f'purpose {purpose}')
    # logg(f'dataset {dataset}')

    # combine args
    if purpose == 'single':
        pass
    elif purpose == 'goodids' and dataset in ['dewac', 'dewiki']:
        file = join(ETL_PATH, f'{dataset}_good_ids.pickle')
    elif purpose == 'lemmap':
        print(dataset)
        file = join(ETL_PATH, f'{dataset}_lemmatization_map.pickle')
    elif purpose == 'embedding':
        file = join(EMB_PATH, dataset, dataset)
    elif purpose in {'topic', 'topics'}:
        # file = join(LDA_PATH, version, corpus_type, 'topics', f'{dataset}_topic-candidates.csv')
        file = join(LDA_PATH, version, corpus_type, 'topics', f'{dataset}_topic-candidates.csv')
    elif purpose in {'score', 'scores'}:
        file = join(LDA_PATH, version, corpus_type, 'topics', f'{dataset}_topic-scores.csv')
    elif purpose == 'wiki_scores':
        file = join(LDA_PATH, version, corpus_type, 'topics', f'{dataset}_topic-wiki-scores.csv')
    elif purpose == 'x2v_scores':
        file = join(LDA_PATH, version, corpus_type, 'topics', f'{dataset}_topic-x2v-scores.csv')
    elif purpose in {'label', 'labels'}:
        file = join(LDA_PATH, version, corpus_type, 'topics', f'{dataset}_label-candidates_full.csv')
    elif purpose == 'dict':
        if dataset == 'dewiki' and 'unfiltered' in args:
            dict_path = join(
                LDA_PATH, version, corpus_type, f'dewiki_noun_{corpus_type}_unfiltered.dict'
            )
        else:
            dict_path = join(LDA_PATH, version, corpus_type, f'{dataset}_{version}_{corpus_type}.dict')
        logg(f'Loading dict from {dict_path}')
        dict_from_corpus = Dictionary.load(dict_path)
        _ = dict_from_corpus[0]  # init dictionary
        return dict_from_corpus
    elif purpose == 'corpus':
        corpus_path = join(LDA_PATH, version, corpus_type, f'{dataset}_{version}_{corpus_type}.mm')
        logg(f'Loading corpus from {corpus_path}')
        corpus = MmCorpus(corpus_path)
        corpus = list(corpus)
        return corpus
    elif purpose == 'texts':
        doc_path = join(LDA_PATH, version, f'{dataset}_{version}_texts.json')
        with open(doc_path, 'r') as fp:
            logg(f'Loading texts from {doc_path}')
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
            logg('oops')
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
        logg(f'Reading {file}')
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
                df = df.set_index('token').text
                df = df[df.str.match(NOUN_PATTERN)]
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
        logg(e)
        if purpose in {'topic', 'topics'}:
            logg('Loading topics via TopicsLoader')
            kwargs = dict(dataset=dataset, version=version, corpus_type=corpus_type, topn=10)
            if params:
                kwargs['param_ids'] = params
            if nbtopics:
                kwargs['nbs_topics'] = nbtopics
            return TopicsLoader(**kwargs).topics


class Unlemmatizer(object):

    def __init__(self):
        self.phrases = load('phrases', 'lemmap')
        self.wiktionary = load('wikt', 'lemmap')

    def unlemmatize_token(self, token, lemmap=None):
        # 1) unlemmatize from original dataset
        if lemmap is not None and token in lemmap:
            word = lemmap[token]

        # 2) unlemmatize from Wikipedia title phrases
        elif token in self.phrases:
            word = self.phrases[token]

        # 3) unlemmatize individual parts of a concatenated token
        elif '_' in token:
            print('unkown phrase', token)
            tokens = token.split('_')
            ts = []
            for t in tokens:
                print(t)
                if t in self.wiktionary:
                    print('token in wikt')
                    print(self.wiktionary.loc[t])
                    ts.append(t)
                elif t.title() in self.wiktionary:
                    print('token.lower in wikt')
                    print(self.wiktionary.loc[t.title()])
                    ts.append(t)
                else:
                    ts.append(t)
            word = '_'.join(ts)
            print(word)

        # 4) nothing to do
        else:
            word = token

        word = word.replace('_.', '.').replace('_', ' ')
        # print('   ', token, '->', word)
        return word

    def unlemmatize_group(self, group):
        lemmap = load(group.name, 'lemmap')
        return group.applymap(lambda x: self.unlemmatize_token(x, lemmap))

    def unlemmatize_topics(self, topics, dataset=None):
        topics = topics.copy()
        if dataset is not None:
            lemmap = load(dataset, 'lemmap')
            topics = topics.applymap(lambda x: self.unlemmatize_token(x, lemmap))
        else:
            topics = topics.groupby('dataset', sort=False).apply(self.unlemmatize_group)
        return topics


# --------------------------------------------------------------------------------------------------
# --- TopicLoader Class ---


class TopicsLoader(object):

    def __init__(
            self, dataset, version='noun', corpus_type='bow',
            param_ids='e42', nbs_topics=100, epochs=30, topn=20,
            filter_bad_terms=False, include_weights=False,
            include_corpus=False, include_texts=False,
            logger=None
    ):
        self.topn = topn
        self.dataset = DSETS.get(dataset, dataset)
        self.version = version
        self.param_ids = [param_ids] if isinstance(param_ids, str) else param_ids
        self.nb_topics_list = [nbs_topics] if isinstance(nbs_topics, int) else nbs_topics
        self.nb_topics = sum(self.nb_topics_list) * len(self.param_ids)
        self.corpus_type = corpus_type
        self.epochs = f"ep{epochs}"
        self.directory = join(LDA_PATH, self.version)
        self.data_filename = f'{self.dataset}_{version}'
        self.filter_terms = filter_bad_terms
        self.include_weights = include_weights
        self.logg = logger.info if logger else print
        self.dictionary = self._load_dict()
        self.topics = self._topn_topics()
        self.corpus = self._load_corpus() if include_corpus else None
        self.texts = self._load_texts() if include_texts else None

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
                topics_weights = []
                for i in range(nb_topics):
                    tokens = []
                    weights = []
                    for term in ldamodel.get_topic_terms(i, topn=self.topn*2):
                        token = ldamodel.id2word[term[0]]
                        weight = term[1]
                        if self.filter_terms and (token in BAD_TOKENS or NOUN_PATTERN.match(token)):
                            continue
                        else:
                            tokens.append(token)
                            weights.append(weight)
                            if len(tokens) == self.topn:
                                break
                    topics.append(tokens)
                    topics_weights.append(weights)

                model_topics = (
                    pd.DataFrame(topics, columns=[f'term{i}' for i in range(self.topn)])
                    .assign(
                        dataset=self.dataset,
                        param_id=param_id,
                        nb_topics=nb_topics
                    )
                )
                if self.include_weights:
                    model_weights = (
                        pd.DataFrame(topics_weights, columns=[f'weight{i}' for i in range(self.topn)])
                    )
                    model_topics = (
                        pd.concat([model_topics, model_weights], axis=1, sort=False)
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
        return self.topics.applymap(lambda x: self.dictionary.token2id[x])

    def _load_model(self, param_id, nb_topics):
        """
        Load an LDA model.
        """
        model_dir = join(self.directory, self.corpus_type, param_id)
        model_file = f'{self.dataset}_LDAmodel_{param_id}_{nb_topics}_{self.epochs}'
        model_path = join(model_dir, model_file)
        self.logg(f'Loading model from {model_path}')
        ldamodel = LdaModel.load(model_path)
        return ldamodel

    def _load_dict(self):
        """
        This dictionary is a different from the model's dict with a different word<->id mapping,
        but from the same corpus and will be used for the Coherence Metrics.
        """
        dict_dir = join(self.directory, self.corpus_type)
        dict_path = join(dict_dir, f'{self.data_filename}_{self.corpus_type}.dict')
        self.logg(f'Loading dictionary from {dict_path}')
        dict_from_corpus: Dictionary = Dictionary.load(dict_path)
        dict_from_corpus.add_documents([[PLACEHOLDER]])
        _ = dict_from_corpus[0]  # init dictionary
        return dict_from_corpus

    def _load_corpus(self):
        """
        load corpus (for u_mass scores)
        """
        corpus_dir = join(self.directory, self.corpus_type)
        corpus_path = join(corpus_dir, f'{self.data_filename}_{self.corpus_type}.mm')
        self.logg(f'Loading corpus from {corpus_path}')
        corpus = MmCorpus(corpus_path)
        corpus = list(corpus)
        corpus.append([(self.dictionary.token2id[PLACEHOLDER], 1.0)])
        return corpus

    def _load_texts(self):
        """
        load texts (for c_... scores using sliding window)
        """
        doc_path = join(self.directory, self.data_filename + '_texts.json')
        with open(doc_path, 'r') as fp:
            self.logg(f'Loading texts from {doc_path}')
            texts = json.load(fp)
        texts.append([PLACEHOLDER])
        return texts


def main():
    tprint(load('topics', 'dewac1', 'e42', 'a42', 100, 25))
    # df = load('dewiki', 'lemmap')
    # tprint(df, 10)
    # dataset = 'dewiki'
    # version = 'noun'
    # corpus_type = 'bow'
    # load(dataset, version, corpus_type, 'dict')
    # load(dataset, version, corpus_type, 'corpus')
    # load(dataset, version, 'texts')

    # dataset = 'news'
    # # param_ids = 'e42'
    #
    # tl = TopicsLoader(
    #     dataset=dataset,
    #     # param_ids=param_ids,
    #     # nbs_topics=nbs_topics,
    #     # version=version,
    #     # topn=nb_candidate_terms
    # )
    # # tprint(tl.topics)
    # ul = Unlemmatizer()
    # topics = ul.unlemmatize_topics(tl.topics)
    # tprint(topics)


if __name__ == '__main__':
    main()


