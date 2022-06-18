import json
import re
from itertools import chain

import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import Doc2Vec, Word2Vec, FastText, LdaModel, LsiModel

from topiclabeling.utils.constants import (
    OUT_DIR,
    NLP_DIR,
    PHRASES_DIR,
    LDA_DIR,
    DATASETS_FULL,
    PARAMS,
    NB_TOPICS,
    METRICS,
    VERSIONS,
    EMB_DIR,
    CORPUS_TYPE,
    WORD_PATTERN,
    BAD_TOKENS,
    PLACEHOLDER,
    LSI_DIR,
    TPX_DIR,
)
from topiclabeling.utils.logging import logg, EXCEPTION


def index_level_dtypes(df):
    return [
        f"{df.index.names[i]}: {df.index.get_level_values(n).dtype}"
        for i, n in enumerate(df.index.names)
    ]


def multiload(dataset, purpose="etl", deprecated=False):
    if dataset.lower().startswith("dewa"):
        dewac = True
    elif dataset.lower().startswith("dewi"):
        dewac = False
    else:
        logg(f"unknown dataset: {dataset}")
        return

    if purpose is not None and purpose.lower() in ["simple", "smpl", "phrase"]:
        if dewac:
            dir_path = PHRASES_DIR / "wiki_phrases"
            pattern = re.compile(r"^dewac_[0-9]{2}_simple_wiki_phrases\.pickle")
            files = sorted([f for f in dir_path.iterdir() if pattern.match(f.name)])
        else:
            dir_path = PHRASES_DIR / "dewiki"
            pattern = re.compile(r"^dewiki_[0-9]+_[0-9]+__[0-9]+_simple\.pickle")
            files = sorted([f for f in dir_path.iterdir() if pattern.match(f.name)])
    elif purpose is not None and purpose.lower() == "nlp":
        dir_path = NLP_DIR
        if dewac:
            pattern = re.compile(r"^dewac_[0-9]{2}_nlp\.pickle")
            files = sorted([f for f in dir_path.iterdir() if pattern.match(f.name)])
        else:
            pattern = re.compile(r"^dewiki_[0-9]+_[0-9]+_nlp\.pickle")
            files = sorted([f for f in dir_path.iterdir() if pattern.match(f.name)])
    else:
        dir_path = OUT_DIR
        if dewac:
            pattern = re.compile(r"^dewac_[0-9]{2}\.pickle")
            files = sorted([f for f in dir_path.iterdir() if pattern.match(f.name)])
        else:
            if deprecated:
                dir_path = dir_path / "deprecated"
                pattern = re.compile(r"^dewiki_[0-9]{2}.*\.pickle\.gz")
                files = sorted([f for f in dir_path.iterdir() if pattern.match(f.name)])
            else:
                files = [dir_path / "dewiki.pickle"]

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
    dataset, version, corpus_type, metrics, params, nbtopics, rerank=False, lsi=False
):
    dfs = []
    tpx_path = LDA_DIR / version / corpus_type / "topics"
    if rerank:
        file = tpx_path / f"{dataset}_reranker-eval"
    elif lsi:
        file = tpx_path / f"{dataset}_lsi_{version}_{corpus_type}_topic-scores"
    else:
        file = tpx_path / f"{dataset}_{version}_{corpus_type}_topic-scores"
    try:
        file = file.with_suffix(".csv")
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
        file = file.parent / f"{file.stem}_germanet.csv"
        logg(f"Reading {file}")
        df = pd.read_csv(file, header=0)
        df = set_index(df)
        df = reduce_df(df, metrics, params, nbtopics)
        dfs.append(df)
    except Exception as e:
        logg(e)
    return pd.concat(dfs, axis=1)


def load(*args):
    """
    work in progress: may not work for all cases, especially not yet for reading distributed
    datasets like dewiki and dewac.
    """

    if not args:
        logg("no arguments, no load")
        return

    single = {
        "hashmap": OUT_DIR / "dewiki_hashmap.pickle",
        "meta": OUT_DIR / "dewiki_metadata.pickle",
        "phrases": OUT_DIR / "dewiki_phrases_lemmatized.pickle",
        "links": OUT_DIR / "dewiki_links.pickle",
        "categories": OUT_DIR / "dewiki_categories.pickle",
        "disamb": OUT_DIR / "dewiki_disambiguation.pickle",
        "wikt": OUT_DIR / "wiktionary_lemmatization_map.pickle",
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
    datasets = (
        list(DATASETS_FULL.keys())
        + list(DATASETS_FULL.values())
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
        elif not dataset and arg in datasets:
            dataset = DATASETS_FULL.get(arg, arg)
        elif not purpose and arg in purposes:
            purpose = arg
        elif not purpose and any(
            [s in arg for s in ["d2v", "w2v", "ftx"] if isinstance(arg, str)]
        ):
            purpose = "embedding"
            dataset = arg
        elif arg in PARAMS:
            params.append(arg)
        elif arg in NB_TOPICS:
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
            df = df[df.str.match(WORD_PATTERN)]
        return df

    # --- good_ideas ---
    elif purpose == "goodids" and dataset in ["dewac", "dewiki"]:
        file = OUT_DIR / f"{dataset}_good_ids.pickle"
        logg(f"Loading {file}")
        return pd.read_pickle(file)

    # --- lemmap ---
    elif purpose == "lemmap":
        file = OUT_DIR / f"{dataset}_lemmatization_map.pickle"
        logg(f"Loading {file}")
        return pd.read_pickle(file)

    # --- embeddings ---
    elif purpose == "embedding":
        file = EMB_DIR / dataset / dataset
        try:
            logg(f"Reading {file}")
            if "d2v" in dataset:
                return Doc2Vec.load(str(file))
            if "w2v" in dataset:
                return Word2Vec.load(str(file))
            if "ftx" in dataset:
                return FastText.load(str(file))
        except Exception as e:
            logg(e, EXCEPTION)

    # --- gensim dict ---
    elif purpose == "dict":
        if dataset == "dewiki" and "unfiltered" in args:
            dict_path = (
                LDA_DIR
                / version
                / corpus_type
                / f"dewiki_noun_{corpus_type}_unfiltered.dict"
            )
        else:
            dict_path = (
                LDA_DIR
                / version
                / corpus_type
                / f"{dataset}_{version}_{corpus_type}.dict"
            )
        try:
            logg(f"Loading dict from {dict_path}")
            # noinspection PyTypeChecker
            dict_from_corpus: Dictionary = Dictionary.load(str(dict_path))
            _ = dict_from_corpus[0]  # init dictionary
            return dict_from_corpus
        except Exception as e:
            logg(e, EXCEPTION)

    # --- MM corpus ---
    elif purpose == "corpus":
        corpus_path = (
            LDA_DIR / version / corpus_type / f"{dataset}_{version}_{corpus_type}.mm"
        )
        try:
            logg(f"Loading corpus from {corpus_path}")
            corpus = MmCorpus(str(corpus_path))
            corpus = list(corpus)
            return corpus
        except Exception as e:
            logg(e, EXCEPTION)

    # --- json texts ---
    elif purpose == "texts":
        doc_path = LDA_DIR / version / f"{dataset}_{version}_texts.json"
        try:
            with open(doc_path, "r") as fp:
                logg(f"Loading texts from {doc_path}")
                texts = json.load(fp)
            return texts
        except Exception as e:
            logg(e, EXCEPTION)

    # --- rerank topics / scores / eval_scores ---
    elif isinstance(purpose, str) and purpose.startswith("rerank"):
        tpx_path = LDA_DIR / version / corpus_type / "topics"
        if purpose.startswith("rerank_score"):
            file = tpx_path / f"{dataset}_reranker-scores.csv"
        elif purpose.startswith("rerank_eval"):
            return load_scores(
                dataset, version, corpus_type, metrics, params, nbtopics, rerank=True
            )
        else:
            file = tpx_path / f"{dataset}_reranker-candidates.csv"
        logg(f"Reading {file}")
        try:
            df = pd.read_csv(file, header=0, index_col=[0, 1, 2, 3, 4])
            df = reduce_df(df, metrics, params, nbtopics)
            return df
        except Exception as e:
            logg(e, EXCEPTION)

    # --- topics ---
    elif purpose in {"topic", "topics"}:
        cols = ["Lemma1", "Lemma2"]
        if dataset in ["gur", "gurevych"]:
            file = OUT_DIR / "gurevych_datasets.csv"
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1])
            return df[cols]
        elif dataset in ["simlex"]:
            file = OUT_DIR / "simlex999.csv"
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1])
            return df[cols]
        elif dataset in ["ws"]:
            file = OUT_DIR / "ws353.csv"
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1])
            return df[cols]
        elif dataset in ["rel", "similarity"]:
            file = OUT_DIR / "similarity_datasets.csv"
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1])
            return df[cols]
        elif dataset in ["survey"]:
            file = TPX_DIR / "survey_topics.csv"
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1, 2, 3])
            survey_cols = [f"term{i}" for i in range(20)]
            return df[survey_cols]

        file = (
            LDA_DIR
            / version
            / corpus_type
            / "topics"
            / f"{dataset}_{version}_{corpus_type}_topic-candidates.csv"
        )
        try:
            df = pd.read_csv(file, header=0)
            logg(f"Reading {file}")
            df = set_index(df)
            return reduce_df(df, metrics, params, nbtopics)
        except:
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
            file_path = LDA_DIR / version / corpus_type / "topics" / dataset
            try:
                file = file_path + "_reranker-candidates.csv"
                df = _load_label_file(file)
            except Exception as e:
                logg(e, EXCEPTION)
        else:
            file_path = (
                LDA_DIR
                / version
                / corpus_type
                / "topics"
                / f"{dataset}_{version}_{corpus_type}"
            )
            df = w2v = None
            if "w2v" in args or "ftx" not in args:
                try:
                    file = file_path + "_label-candidates.csv"
                    df = w2v = _load_label_file(file)
                except Exception as e:
                    logg(e, EXCEPTION)
            if "ftx" in args or "w2v" not in args:
                try:
                    file = file_path + "_label-candidates_ftx.csv"
                    df = ftx = _load_label_file(file)
                    if w2v is not None:
                        ftx = ftx.query('label_method != "d2v"')
                        df = w2v.append(ftx).sort_index()
                except Exception as e:
                    logg(e, EXCEPTION)
        return df

    # --- scores ---
    elif purpose in {"score", "scores"}:
        if "lsi" in args:
            return load_scores(
                dataset, version, corpus_type, metrics, params, nbtopics, lsi=True
            )
        elif "rerank" in args:
            return load_scores(
                dataset, version, corpus_type, metrics, params, nbtopics, rerank=True
            )
        else:
            return load_scores(dataset, version, corpus_type, metrics, params, nbtopics)

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
            file = OUT_DIR / "gurevych_datasets.csv"
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1])
            return df
        elif dataset in ["simlex"]:
            file = OUT_DIR / "simlex999.csv"
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1])
            return df
        elif dataset in ["ws"]:
            file = OUT_DIR / "ws353.csv"
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1])
            return df
        elif dataset in ["rel", "similarity"]:
            file = OUT_DIR / "similarity_datasets.csv"
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1])
            return df
        elif dataset in ["survey"]:
            file = TPX_DIR / "survey_topics.csv"
            logg(f"Reading {file}")
            df = pd.read_csv(file, header=0, index_col=[0, 1, 2, 3])
            return df

        if purpose in {"etl", None}:
            directory = OUT_DIR
            suffix = ""
        elif purpose == "nlp":
            directory = NLP_DIR
            suffix = "_nlp"
        elif purpose in {"simple", "smpl"}:
            directory = PHRASES_DIR
            suffix = "_simple"
        elif purpose in {"wiki", "wiki_phrases", "phrases"}:
            directory = PHRASES_DIR / "wiki_phrases"
            suffix = "_simple_wiki_phrases"
        else:
            logg("oops")
            return

        if dataset == "speeches":
            file = [
                directory / f'{DATASETS_FULL["E"]}{suffix}.pickle',
                directory / f'{DATASETS_FULL["P"]}{suffix}.pickle',
            ]
        elif dataset == "news":
            file = [
                directory / f'{DATASETS_FULL["FA"]}{suffix}.pickle',
                directory / f'{DATASETS_FULL["FO"]}{suffix}.pickle',
            ]
        elif dataset == "dewac1":
            file = directory / f'{dataset.replace("dewac1", "dewac_01")}{suffix}.pickle'
        elif dataset in {"dewac", "dewiki"}:
            dfs = [d for d in multiload(dataset, purpose, deprecated)]
            return pd.concat(dfs)
        else:
            if purpose in {"etl", None} and deprecated and dataset in {"FAZ", "FOCUS"}:
                directory = OUT_DIR / "deprecated"
                if dataset == "FAZ":
                    file = [directory / "FAZ.pickle.gz", directory / "FAZ2.pickle.gz"]
                else:
                    file = directory / "FOCUS.pickle.gz"
            else:
                file = directory / f"{dataset}{suffix}.pickle"
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
            logg(e, EXCEPTION)


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
            logg(f"unknown phrase {token}")
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
            logg(f"   {token} -> {word}")

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
    ):
        self.dataset = DATASETS_FULL.get(dataset, dataset)
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
        self.directory = LDA_DIR / self.version
        self.data_filename = f"{self.dataset}_{version}"
        self.filter_terms = filter_bad_terms
        self.include_weights = include_weights
        self.column_names_terms = [f"term{i}" for i in range(self.topn)]
        self.column_names_weights = [f"weight{i}" for i in range(self.topn)]
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
                            token in BAD_TOKENS or WORD_PATTERN.match(token)
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
            model_dir = LSI_DIR / self.version / self.corpus_type
            model_file = f"{self.dataset}_LSImodel_{nb_topics}"
            model_path = model_dir / model_file
            model = LsiModel.load(str(model_path))
        else:
            model_dir = self.directory / self.corpus_type / param_id
            model_file = f"{self.dataset}_LDAmodel_{param_id}_{nb_topics}_{self.epochs}"
            model_path = model_dir / model_file
            model = LdaModel.load(str(model_path))
        logg(f"Loading model from {model_path}")

        return model

    def _load_dict(self):
        """
        This dictionary is a different from the model's dict with a different word<->id mapping,
        but from the same corpus and will be used for the Coherence Metrics.
        """
        dict_dir = self.directory / self.corpus_type
        dict_path = dict_dir / f"{self.data_filename}_{self.corpus_type}.dict"
        logg(f"Loading dictionary from {dict_path}")
        # noinspection PyTypeChecker
        dict_from_corpus: Dictionary = Dictionary.load(str(dict_path))
        dict_from_corpus.add_documents([[PLACEHOLDER]])
        _ = dict_from_corpus[0]  # init dictionary

        return dict_from_corpus

    def _load_corpus(self):
        """
        Load corpus (for u_mass scores)
        """
        corpus_dir = self.directory / self.corpus_type
        corpus_path = corpus_dir / f"{self.data_filename}_{self.corpus_type}.mm"
        logg(f"Loading corpus from {corpus_path}")
        corpus = MmCorpus(corpus_path)
        corpus = list(corpus)
        corpus.append([(self.dictionary.token2id[PLACEHOLDER], 1.0)])

        return corpus

    def _load_texts(self):
        """
        Load texts (for c_... scores using sliding window)
        """
        doc_path = self.directory / self.data_filename + "_texts.json"
        with open(doc_path, "r") as fp:
            logg(f"Loading texts from {doc_path}")
            texts = json.load(fp)
        texts.append([PLACEHOLDER])

        return texts
