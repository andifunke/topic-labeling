import json
import logging
import re
import sys
from genericpath import exists
from itertools import chain
from os import makedirs, listdir
from os.path import join
from pprint import pformat
import warnings

import gensim
import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import Doc2Vec, Word2Vec, FastText, LdaModel, LsiModel
from pandas.errors import DtypeWarning

from constants import (
    ETL_PATH,
    NLP_PATH,
    SMPL_PATH,
    LDA_PATH,
    DSETS,
    PARAMS,
    NBTOPICS,
    METRICS,
    VERSIONS,
    EMB_PATH,
    CORPUS_TYPE,
    NOUN_PATTERN,
    BAD_TOKENS,
    PLACEHOLDER,
    LSI_PATH,
    TPX_PATH,
)

try:
    from tabulate import tabulate
except ImportError as ie:
    print(ie)

warnings.simplefilter(action="ignore", category=DtypeWarning)


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
        kwargs["floatfmt"] = floatfmt
    try:
        print(
            tabulate(df, headers="keys", tablefmt="pipe", showindex="always", **kwargs)
        )
    except:
        print(df)
    print("shape:", shape, "\n")

    if to_latex:
        print(df.to_latex(bold_rows=True))


def index_level_dtypes(df):
    return [
        f"{df.index.names[i]}: {df.index.get_level_values(n).dtype}"
        for i, n in enumerate(df.index.names)
    ]


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


def init_logging(
    name="", basic=True, to_stdout=False, to_file=True, log_file=None, log_dir="../logs"
):

    if log_file is None:
        log_file = name + ".log" if name else "train.log"

    if basic:
        if to_file:
            if not exists(log_dir):
                makedirs(log_dir)
            file_path = join(log_dir, log_file)
            logging.basicConfig(
                filename=file_path,
                format="%(asctime)s - %(name)s - %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                level=logging.INFO,
            )
        else:
            logging.basicConfig(
                format="%(asctime)s - %(name)s - %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                level=logging.INFO,
            )
        logger = logging.getLogger()

    else:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
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

    logger.info("")
    logger.info("#" * 50)
    logger.info("----- %s -----" % name.upper())
    logger.info("----- start -----")
    logger.info("python: " + sys.version.replace("\n", " "))
    logger.info("pandas: " + pd.__version__)
    logger.info("gensim: " + gensim.__version__)

    return logger


def log_args(logger, args):
    logger.info("\n" + pformat(vars(args)))


def multiload(dataset, purpose="etl", deprecated=False):
    if dataset.lower().startswith("dewa"):
        dewac = True
    elif dataset.lower().startswith("dewi"):
        dewac = False
    else:
        print("unkown dataset")
        return

    if purpose is not None and purpose.lower() in ["simple", "smpl", "phrase"]:
        if dewac:
            dpath = join(SMPL_PATH, "wiki_phrases")
            pattern = re.compile(r"^dewac_[0-9]{2}_simple_wiki_phrases\.pickle")
            files = sorted([join(dpath, f) for f in listdir(dpath) if pattern.match(f)])
        else:
            dpath = join(SMPL_PATH, "dewiki")
            pattern = re.compile(r"^dewiki_[0-9]+_[0-9]+__[0-9]+_simple\.pickle")
            files = sorted([join(dpath, f) for f in listdir(dpath) if pattern.match(f)])
    elif purpose is not None and purpose.lower() == "nlp":
        dpath = NLP_PATH
        if dewac:
            pattern = re.compile(r"^dewac_[0-9]{2}_nlp\.pickle")
            files = sorted([join(dpath, f) for f in listdir(dpath) if pattern.match(f)])
        else:
            pattern = re.compile(r"^dewiki_[0-9]+_[0-9]+_nlp\.pickle")
            files = sorted([join(dpath, f) for f in listdir(dpath) if pattern.match(f)])
    else:
        dpath = ETL_PATH
        if dewac:
            pattern = re.compile(r"^dewac_[0-9]{2}\.pickle")
            files = sorted([join(dpath, f) for f in listdir(dpath) if pattern.match(f)])
        else:
            if deprecated:
                dpath = join(dpath, "deprecated")
                pattern = re.compile(r"^dewiki_[0-9]{2}.*\.pickle\.gz")
                files = sorted(
                    [join(dpath, f) for f in listdir(dpath) if pattern.match(f)]
                )
            else:
                files = [join(dpath, "dewiki.pickle")]

    length = len(files)
    for i, file in enumerate(files, 1):
        print(f"Reading {i:02d}/{length}: {file}")
        yield pd.read_pickle(file)


def reduce_df(df, metrics, params, nbtopics):
    if len(metrics) > 0:
        try:
            df = df.query("metric in @metrics")
        except Exception as e:
            print(e)
    if len(params) > 0:
        try:
            df = df.query("param_id in @params")
        except Exception as e:
            print(e)
    if len(nbtopics) > 0:
        try:
            df = df.query("nb_topics in @nbtopics")
        except Exception as e:
            print(e)
    return df


def flatten_columns(df):
    df = pd.DataFrame(df.to_records())

    def rename_column(col):
        if col.startswith("("):
            col = eval(col)
            if col[0] == "score":
                col = col[1]
            else:
                col = "_".join(col)
        return col

    df = df.rename(columns=rename_column)
    df = set_index(df)
    return df


def set_index(df):
    keys = [
        key
        for key in [
            "dataset",
            "param_id",
            "nb_topics",
            "topic_idx",
            "label_method",
            "metric",
        ]
        if key in df.columns
    ]
    df = df.set_index(keys)
    return df


def load_scores(
    dataset,
    version,
    corpus_type,
    metrics,
    params,
    nbtopics,
    logg=print,
    rerank=False,
    lsi=False,
):
    dfs = []
    tpx_path = join(LDA_PATH, version, corpus_type, "topics")
    if rerank:
        file_prefix = join(tpx_path, f"{dataset}_reranker-eval")
    elif lsi:
        file_prefix = join(
            tpx_path, f"{dataset}_lsi_{version}_{corpus_type}_topic-scores"
        )
    else:
        file_prefix = join(tpx_path, f"{dataset}_{version}_{corpus_type}_topic-scores")
    try:
        file = file_prefix + ".csv"
        logg(f"Reading {file}")
        df = pd.read_csv(file, header=[0, 1], skipinitialspace=True)
        cols = list(df.columns)
        for column in cols:
            if column[0].startswith("Unnamed"):
                col_name = df.loc[0, column]
                df[col_name] = df[column]
                df = df.drop(column, axis=1)
        df = df.drop(0)
        if "nb_topics" in df.columns:
            df.nb_topics = df.nb_topics.astype(int)
        if "topic_idx" in df.columns:
            df.topic_idx = df.topic_idx.astype(int)
        df = df.drop(["stdev", "support"], level=0, axis=1)
        df = set_index(df)
        df = flatten_columns(df)
        df = reduce_df(df, metrics, params, nbtopics)
        dfs.append(df)
    except Exception as e:
        logg(e)
    try:
        file = file_prefix + "_germanet.csv"
        logg(f"Reading {file}")
        df = pd.read_csv(file, header=0)
        df = set_index(df)
        df = reduce_df(df, metrics, params, nbtopics)
        dfs.append(df)
    except Exception as e:
        logg(e)
    return pd.concat(dfs, axis=1)


def load(*args, logger=None, logg=print):
    """
    work in progress: may not work for all cases, especially not yet for reading distributed
    datsets like dewiki and dewac.
    """

    logg = logger.info if logger else logg

    if not args:
        logg("no arguments, no load")
        return

    single = {
        "hashmap": join(ETL_PATH, "dewiki_hashmap.pickle"),
        "meta": join(ETL_PATH, "dewiki_metadata.pickle"),
        "phrases": join(ETL_PATH, "dewiki_phrases_lemmatized.pickle"),
        "links": join(ETL_PATH, "dewiki_links.pickle"),
        "categories": join(ETL_PATH, "dewiki_categories.pickle"),
        "disamb": join(ETL_PATH, "dewiki_disambiguation.pickle"),
        "wikt": join(ETL_PATH, "wiktionary_lemmatization_map.pickle"),
    }
    dataset = None
    purposes = {
        "goodids",
        "etl",
        "nlp",
        "simple",
        "smpl",
        "wiki_phrases",
        "embedding",
        "topic",
        "topics",
        "label",
        "labels",
        "lda",
        "ldamodel",
        "score",
        "scores",
        "lemmap",
        "disamb",
        "dict",
        "corpus",
        "texts",
        "wiki_scores",
        "x2v_scores",
        "rerank",
        "rerank_score",
        "rerank_scores",
        "rerank_eval",
    }
    purpose = None
    version = None
    corpus_type = None
    params = []
    nbtopics = []
    metrics = []
    deprecated = False
    dsets = (
        list(DSETS.keys())
        + list(DSETS.values())
        + ["gurevych", "gur", "simlex", "ws", "rel", "similarity", "survey"]
    )

    if isinstance(args, str):
        args = [args]
    args = [arg.replace("-", "_") if isinstance(arg, str) else arg for arg in args]

    # --- parse args ---
    for arg in args:
        arg = arg.lower() if isinstance(arg, str) else arg
        if arg in single:
            if arg == "phrases" and "lemmap" in args:
                dataset = "dewiki_phrases"
                purpose = "lemmap"
            else:
                purpose = "single"
                dataset = arg
                break
        elif not dataset and arg in dsets:
            dataset = DSETS.get(arg, arg)
        elif not purpose and arg in purposes:
            purpose = arg
        elif not purpose and any(
            [s in arg for s in ["d2v", "w2v", "ftx"] if isinstance(arg, str)]
        ):
            purpose = "embedding"
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
        elif arg == "deprecated":
            deprecated = True

    # --- setting default values ---
    if version is None:
        version = "noun"
    if corpus_type is None:
        corpus_type = "bow"
    if "default" in args:
        params.append("e42")
        nbtopics.append("100")
        metrics.append("ref")

    # --- single ---
    if purpose == "single":
        df = pd.read_pickle(single[dataset])
        if "phrases" in args and "minimal" in args:
            df = df.set_index("token").text
            df = df[df.str.match(NOUN_PATTERN)]
        return df

    # --- good_ideas ---
    elif purpose == "goodids" and dataset in ["dewac", "dewiki"]:
        file = join(ETL_PATH, f"{dataset}_good_ids.pickle")
        logg(f"Loading {file}")
        return pd.read_pickle(file)

    # --- lemmap ---
    elif purpose == "lemmap":
        file = join(ETL_PATH, f"{dataset}_lemmatization_map.pickle")
        logg(f"Loading {file}")
        return pd.read_pickle(file)

    # --- embeddings ---
    elif purpose == "embedding":
        file = join(EMB_PATH, dataset, dataset)
        try:
            logg(f"Reading {file}")
            if "d2v" in dataset:
                return Doc2Vec.load(file)
            if "w2v" in dataset:
                return Word2Vec.load(file)
            if "ftx" in dataset:
                return FastText.load(file)
        except Exception as e:
            logg(e)

    # --- gensim dict ---
    elif purpose == "dict":
        if dataset == "dewiki" and "unfiltered" in args:
            dict_path = join(
                LDA_PATH,
                version,
                corpus_type,
                f"dewiki_noun_{corpus_type}_unfiltered.dict",
            )
        else:
            dict_path = join(
                LDA_PATH,
                version,
                corpus_type,
                f"{dataset}_{version}_{corpus_type}.dict",
            )
        try:
            logg(f"Loading dict from {dict_path}")
            dict_from_corpus = Dictionary.load(dict_path)
            _ = dict_from_corpus[0]  # init dictionary
            return dict_from_corpus
        except Exception as e:
            logg(e)

    # --- MM corpus ---
    elif purpose == "corpus":
        corpus_path = join(
            LDA_PATH, version, corpus_type, f"{dataset}_{version}_{corpus_type}.mm"
        )
        try:
            logg(f"Loading corpus from {corpus_path}")
            corpus = MmCorpus(corpus_path)
            corpus = list(corpus)
            return corpus
        except Exception as e:
            logg(e)

    # --- json texts ---
    elif purpose == "texts":
        doc_path = join(LDA_PATH, version, f"{dataset}_{version}_texts.json")
        try:
            with open(doc_path, "r") as fp:
                logg(f"Loading texts from {doc_path}")
                texts = json.load(fp)
            return texts
        except Exception as e:
            logg(e)

    # --- rerank topics / scores / eval_scores ---
    elif isinstance(purpose, str) and purpose.startswith("rerank"):
        tpx_path = join(LDA_PATH, version, corpus_type, "topics")
        if purpose.startswith("rerank_score"):
            file = join(tpx_path, f"{dataset}_reranker-scores.csv")
        elif purpose.startswith("rerank_eval"):
            return load_scores(
                dataset,
                version,
                corpus_type,
                metrics,
                params,
                nbtopics,
                logg=logg,
                rerank=True,
            )
        else:
            file = join(tpx_path, f"{dataset}_reranker-candidates.csv")
        logg(f"Reading {file}")
        try:
            df = pd.read_csv(file, header=0, index_col=[0, 1, 2, 3, 4])
            df = reduce_df(df, metrics, params, nbtopics)
            return df
        except Exception as e:
            logg(e)

    # --- topics ---
    elif purpose in {"topic", "topics"}:
        cols = ["Lemma1", "Lemma2"]
        if dataset in ["gur", "gurevych"]:
            file = join(ETL_PATH, "gurevych_datasets.csv")
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1])
            return df[cols]
        elif dataset in ["simlex"]:
            file = join(ETL_PATH, "simlex999.csv")
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1])
            return df[cols]
        elif dataset in ["ws"]:
            file = join(ETL_PATH, "ws353.csv")
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1])
            return df[cols]
        elif dataset in ["rel", "similarity"]:
            file = join(ETL_PATH, "similarity_datasets.csv")
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1])
            return df[cols]
        elif dataset in ["survey"]:
            file = join(TPX_PATH, "survey_topics.csv")
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1, 2, 3])
            survey_cols = [f"term{i}" for i in range(20)]
            return df[survey_cols]

        file = join(
            LDA_PATH,
            version,
            corpus_type,
            "topics",
            f"{dataset}_{version}_{corpus_type}_topic-candidates.csv",
        )
        try:
            df = pd.read_csv(file, header=0)
            logg(f"Reading {file}")
            df = set_index(df)
            return reduce_df(df, metrics, params, nbtopics)
        except Exception as e:
            # logg(e)
            # logg('Loading topics via TopicsLoader')
            lsi = "lsi" in args
            kwargs = dict(
                dataset=dataset,
                version=version,
                corpus_type=corpus_type,
                topn=10,
                lsi=lsi,
            )
            if params:
                kwargs["param_ids"] = params
            if nbtopics:
                kwargs["nbs_topics"] = nbtopics
            return TopicsLoader(**kwargs).topics

    # --- labels ---
    elif purpose in {"label", "labels"}:

        def _load_label_file(file_):
            logg(f"Reading {file_}")
            df_ = pd.read_csv(file_, header=0)
            df_ = set_index(df_)
            df_ = df_.applymap(eval)
            if "minimal" in args:
                df_ = df_.query('label_method in ["comb", "comb_ftx"]').applymap(
                    lambda x: x[0]
                )
            return reduce_df(df_, metrics, params, nbtopics)

        df = None
        if "rerank" in args:
            fpath = join(LDA_PATH, version, corpus_type, "topics", dataset)
            try:
                file = fpath + "_reranker-candidates.csv"
                df = _load_label_file(file)
            except Exception as e:
                logg(e)
        else:
            fpath = join(
                LDA_PATH,
                version,
                corpus_type,
                "topics",
                f"{dataset}_{version}_{corpus_type}",
            )
            df = w2v = None
            if "w2v" in args or "ftx" not in args:
                try:
                    file = fpath + "_label-candidates.csv"
                    df = w2v = _load_label_file(file)
                except Exception as e:
                    logg(e)
            if "ftx" in args or "w2v" not in args:
                try:
                    file = fpath + "_label-candidates_ftx.csv"
                    df = ftx = _load_label_file(file)
                    if w2v is not None:
                        ftx = ftx.query('label_method != "d2v"')
                        df = w2v.append(ftx).sort_index()
                except Exception as e:
                    logg(e)
        return df

    # --- scores ---
    elif purpose in {"score", "scores"}:
        if "lsi" in args:
            return load_scores(
                dataset,
                version,
                corpus_type,
                metrics,
                params,
                nbtopics,
                lsi=True,
                logg=logg,
            )
        elif "rerank" in args:
            return load_scores(
                dataset,
                version,
                corpus_type,
                metrics,
                params,
                nbtopics,
                rerank=True,
                logg=logg,
            )
        else:
            return load_scores(
                dataset, version, corpus_type, metrics, params, nbtopics, logg=logg
            )

    # --- pipelines ---
    elif purpose in {
        "nlp",
        "simple",
        "smpl",
        "wiki",
        "wiki_phrases",
        "phrases",
        "etl",
        None,
    }:
        if dataset in ["gur", "gurevych"]:
            file = join(ETL_PATH, "gurevych_datasets.csv")
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1])
            return df
        elif dataset in ["simlex"]:
            file = join(ETL_PATH, "simlex999.csv")
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1])
            return df
        elif dataset in ["ws"]:
            file = join(ETL_PATH, "ws353.csv")
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1])
            return df
        elif dataset in ["rel", "similarity"]:
            file = join(ETL_PATH, "similarity_datasets.csv")
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1])
            return df
        elif dataset in ["survey"]:
            file = join(TPX_PATH, "survey_topics.csv")
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1, 2, 3])
            return df

        if purpose in {"etl", None}:
            directory = ETL_PATH
            suffix = ""
        elif purpose == "nlp":
            directory = NLP_PATH
            suffix = "_nlp"
        elif purpose in {"simple", "smpl"}:
            directory = SMPL_PATH
            suffix = "_simple"
        elif purpose in {"wiki", "wiki_phrases", "phrases"}:
            directory = join(SMPL_PATH, "wiki_phrases")
            suffix = "_simple_wiki_phrases"
        else:
            logg("oops")
            return
        if dataset == "speeches":
            file = [
                join(directory, f'{DSETS["E"]}{suffix}.pickle'),
                join(directory, f'{DSETS["P"]}{suffix}.pickle'),
            ]
        elif dataset == "news":
            file = [
                join(directory, f'{DSETS["FA"]}{suffix}.pickle'),
                join(directory, f'{DSETS["FO"]}{suffix}.pickle'),
            ]
        elif dataset == "dewac1":
            file = join(
                directory, f'{dataset.replace("dewac1", "dewac_01")}{suffix}.pickle'
            )
        elif dataset in {"dewac", "dewiki"}:
            dfs = [d for d in multiload(dataset, purpose, deprecated)]
            return pd.concat(dfs)
        else:
            if purpose in {"etl", None} and deprecated and dataset in {"FAZ", "FOCUS"}:
                directory = join(ETL_PATH, "deprecated")
                if dataset == "FAZ":
                    file = [
                        join(directory, "FAZ.pickle.gz"),
                        join(directory, "FAZ2.pickle.gz"),
                    ]
                else:
                    file = join(directory, "FOCUS.pickle.gz")
            else:
                file = join(directory, f"{dataset}{suffix}.pickle")
        try:
            logg(f"Reading {file}")
            if isinstance(file, str):
                return pd.read_pickle(file)
            else:
                return pd.concat(
                    [
                        pd.read_pickle(f).drop("new", axis=1, errors="ignore")
                        for f in file
                    ]
                )
        except Exception as e:
            logg(e)


class Unlemmatizer(object):
    def __init__(self):
        self.phrases = load("phrases", "lemmap")
        self.wiktionary = load("wikt", "lemmap")

    def unlemmatize_token(self, token, lemmap=None):
        # 1) unlemmatize from Wikipedia title phrases
        if token in self.phrases:
            word = self.phrases[token]

        # 2) unlemmatize from original dataset
        elif lemmap is not None and token in lemmap:
            word = lemmap[token]

        # 3) unlemmatize individual parts of a concatenated token
        elif "_" in token:
            print("unkown phrase", token)
            tokens = token.split("_")
            ts = []
            for t in tokens:
                print(t)
                if t in self.wiktionary:
                    print("token in wikt")
                    print(self.wiktionary.loc[t])
                    ts.append(t)
                elif t.title() in self.wiktionary:
                    print("token.lower in wikt")
                    print(self.wiktionary.loc[t.title()])
                    ts.append(t)
                else:
                    ts.append(t.title())
            word = "_".join(ts)

        # 4) nothing to do
        else:
            word = token

        word = word.replace("_.", ".").replace("_", " ")
        if word != token:
            print("   ", token, "->", word)
        return word

    def unlemmatize_group(self, group):
        lemmap = load(group.name, "lemmap")
        return group.applymap(lambda x: self.unlemmatize_token(x, lemmap))

    def unlemmatize_topics(self, topics, dataset=None):
        topics = topics.copy()
        if dataset is not None:
            lemmap = load(dataset, "lemmap")
            topics = topics.applymap(lambda x: self.unlemmatize_token(x, lemmap))
        else:
            topics = topics.groupby("dataset", sort=False).apply(self.unlemmatize_group)
        return topics

    def unlemmatize_labels(self, labels):
        labels = labels.copy()
        labels = labels.applymap(self.unlemmatize_token)
        return labels


# --------------------------------------------------------------------------------------------------
# --- TopicLoader Class ---


class TopicsLoader(object):
    def __init__(
        self,
        dataset,
        version="noun",
        corpus_type="bow",
        param_ids="e42",
        nbs_topics=100,
        epochs=30,
        topn=20,
        lsi=False,
        filter_bad_terms=False,
        include_weights=False,
        include_corpus=False,
        include_texts=False,
        logger=None,
        logg=print,
    ):
        self.dataset = DSETS.get(dataset, dataset)
        self.version = version
        self.param_ids = [param_ids] if isinstance(param_ids, str) else param_ids
        self.nb_topics_list = (
            [nbs_topics] if isinstance(nbs_topics, int) else nbs_topics
        )
        self.nb_topics = sum(self.nb_topics_list) * len(self.param_ids)
        self.corpus_type = corpus_type
        self.epochs = f"ep{epochs}"
        self.topn = topn
        self.lsi = lsi
        self.directory = join(LDA_PATH, self.version)
        self.data_filename = f"{self.dataset}_{version}"
        self.filter_terms = filter_bad_terms
        self.include_weights = include_weights
        self.column_names_terms = [f"term{i}" for i in range(self.topn)]
        self.column_names_weights = [f"weight{i}" for i in range(self.topn)]
        self.logg = logger.info if logger else logg
        self.dictionary = self._load_dict()
        self.topics = self._topn_topics()
        self.corpus = self._load_corpus() if include_corpus else None
        self.texts = self._load_texts() if include_texts else None

    def _topn_topics(self):
        """
        get the topn topics from the LDA/LSI-model in DataFrame format
        """
        if self.lsi:
            columns = self.column_names_terms + self.column_names_weights
            dfs = []
            for nb_topics in self.nb_topics_list:
                model = self._load_model(None, nb_topics)
                topics = model.show_topics(num_words=self.topn, formatted=False)
                topics = [list(chain(*zip(*topic[1]))) for topic in topics]
                df = pd.DataFrame(topics, columns=columns)
                df["nb_topics"] = nb_topics
                df["topic_idx"] = df.index.values
                dfs.append(df)
            df = pd.concat(dfs)
            df["dataset"] = self.dataset
            df["param_id"] = "lsi"
            df = df.set_index(["dataset", "param_id", "nb_topics", "topic_idx"])
            if not self.include_weights:
                df = df.loc[:, "term0":f"term{self.topn-1}"]
            return df

        all_topics = []
        for param_id in self.param_ids:
            for nb_topics in self.nb_topics_list:
                model = self._load_model(param_id, nb_topics)
                # topic building ignoring placeholder values
                topics = []
                topics_weights = []
                for i in range(nb_topics):
                    tokens = []
                    weights = []
                    for term in model.get_topic_terms(i, topn=self.topn * 2):
                        token = model.id2word[term[0]]
                        weight = term[1]
                        if self.filter_terms and (
                            token in BAD_TOKENS or NOUN_PATTERN.match(token)
                        ):
                            continue
                        else:
                            tokens.append(token)
                            weights.append(weight)
                            if len(tokens) == self.topn:
                                break
                    topics.append(tokens)
                    topics_weights.append(weights)

                model_topics = pd.DataFrame(
                    topics, columns=self.column_names_terms
                ).assign(dataset=self.dataset, param_id=param_id, nb_topics=nb_topics)
                if self.include_weights:
                    model_weights = pd.DataFrame(
                        topics_weights, columns=self.column_names_weights
                    )
                    model_topics = pd.concat(
                        [model_topics, model_weights], axis=1, sort=False
                    )
                all_topics.append(model_topics)
        topics = (
            pd.concat(all_topics)
            .rename_axis("topic_idx")
            .reset_index(drop=False)
            .set_index(["dataset", "param_id", "nb_topics", "topic_idx"])
        )
        return topics

    def topic_ids(self):
        return self.topics[self.column_names_terms].applymap(
            lambda x: self.dictionary.token2id[x]
        )

    def _load_model(self, param_id, nb_topics):
        """
        Load an LDA model.
        """
        if self.lsi:
            model_dir = join(LSI_PATH, self.version, self.corpus_type)
            model_file = f"{self.dataset}_LSImodel_{nb_topics}"
            model_path = join(model_dir, model_file)
            model = LsiModel.load(model_path)
        else:
            model_dir = join(self.directory, self.corpus_type, param_id)
            model_file = f"{self.dataset}_LDAmodel_{param_id}_{nb_topics}_{self.epochs}"
            model_path = join(model_dir, model_file)
            model = LdaModel.load(model_path)
        self.logg(f"Loading model from {model_path}")
        return model

    def _load_dict(self):
        """
        This dictionary is a different from the model's dict with a different word<->id mapping,
        but from the same corpus and will be used for the Coherence Metrics.
        """
        dict_dir = join(self.directory, self.corpus_type)
        dict_path = join(dict_dir, f"{self.data_filename}_{self.corpus_type}.dict")
        self.logg(f"Loading dictionary from {dict_path}")
        dict_from_corpus: Dictionary = Dictionary.load(dict_path)
        dict_from_corpus.add_documents([[PLACEHOLDER]])
        _ = dict_from_corpus[0]  # init dictionary
        return dict_from_corpus

    def _load_corpus(self):
        """
        load corpus (for u_mass scores)
        """
        corpus_dir = join(self.directory, self.corpus_type)
        corpus_path = join(corpus_dir, f"{self.data_filename}_{self.corpus_type}.mm")
        self.logg(f"Loading corpus from {corpus_path}")
        corpus = MmCorpus(corpus_path)
        corpus = list(corpus)
        corpus.append([(self.dictionary.token2id[PLACEHOLDER], 1.0)])
        return corpus

    def _load_texts(self):
        """
        load texts (for c_... scores using sliding window)
        """
        doc_path = join(self.directory, self.data_filename + "_texts.json")
        with open(doc_path, "r") as fp:
            self.logg(f"Loading texts from {doc_path}")
            texts = json.load(fp)
        texts.append([PLACEHOLDER])
        return texts


def main():
    # tprint(load('topics', 'gur'))
    # topics = TopicsLoader('O', nbs_topics=[10, 25, 50, 100], lsi=True, topn=10).topics
    # tprint(load('score', 'O'), 50)

    # for x in load('phrases'):
    #     print(x)
    # tprint(load('dewac1', 'topics', 'lsi', 10, 25))
    tprint(
        load("dewac1", "labels", "rerank", "e42", 100)
    )  # .query('metric == "w2v_matches"'))
    # from itertools import islice
    # for d in islice(load('dewik'), 2):
    #     tprint(d, 2)


if __name__ == "__main__":
    main()
