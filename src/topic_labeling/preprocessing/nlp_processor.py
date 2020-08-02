# -*- coding: utf-8 -*-
import csv
import gc
from os import makedirs
from os.path import exists, join
from time import time

import pandas as pd
import spacy
from tqdm import tqdm

from topic_labeling.utils.constants import (
    VOC_DIR, TEXT, LEMMA, IWNLP, POS, TOK_IDX, SENT_START, ENT_IOB, ENT_TYPE, ENT_IDX, TOKEN,
    SENT_IDX, HASH, NOUN_PHRASE, NLP_DIR, PUNCT, TITLE, ETL_DIR, DESCRIPTION, IWNLP_FILE
)
from topic_labeling.preprocessing.nlp_lemmatizer_plus import LemmatizerPlus


class NLProcessor(object):

    FIELDS = [HASH, TOK_IDX, SENT_IDX, TEXT, TOKEN, POS, ENT_IOB, ENT_IDX, ENT_TYPE, NOUN_PHRASE]

    def __init__(self, spacy_path, lemmatizer_path=IWNLP_FILE, logg=None):
        self.logg = logg if logg else print

        # ------ load spacy and iwnlp ------
        logg("loading spacy")
        self.nlp = spacy.load(spacy_path)
        # nlp = spacy.load(de, disable=['parser'])   # <-- load without dependency parser (fast)

        if exists(VOC_DIR):
            logg(f"reading vocab from {VOC_DIR}")
            self.nlp.vocab.from_disk(VOC_DIR)

        logg("loading IWNLPWrapper")
        self.lemmatizer = LemmatizerPlus(lemmatizer_path, self.nlp)
        self.nlp.add_pipe(self.lemmatizer)
        self.stringstore = self.nlp.vocab.strings

    def read_process_store(
            self, file_path, corpus_name, store=True, vocab_to_disk=False, start=0, stop=None,
    ):
        """
        TODO

        :param file_path:
        :param corpus_name:
        :param store:
        :param vocab_to_disk:
        :param start:
        :param stop:
        :return:
        """

        logg = self.logg
        logg(f"*** start new corpus: {corpus_name}")
        t0 = time()

        # read the etl dataframe
        slicing = f"[{start:d}:{stop:d}]" if (start or stop) else ''
        logg(f"{corpus_name}: reading corpus{slicing} from {file_path}")
        df = self.read(file_path, start=start, stop=stop)

        logg(f"collect: {gc.collect()}")

        # start the nlp pipeline
        logg(f"{corpus_name}: start processing")
        # self.check_docs(df); return
        reader = self.process_docs(df)

        if store:
            if start or stop:
                suffix = f'_{start:d}_{stop-1:d}_nlp'
            else:
                suffix = '_nlp'

            makedirs(NLP_DIR, exist_ok=True)
            filename = NLP_DIR / f'{corpus_name}{suffix}.csv'
            self.logg(f"{corpus_name}: saving to {filename}")

            with open(filename, 'w') as fp:
                header = True
                for doc in reader:
                    doc.to_csv(
                        fp, mode='a',
                        sep='\t', quoting=csv.QUOTE_NONE,
                        index=None, header=header
                    )
                    header = False

        if vocab_to_disk:
            # stored with each corpus, in case anythings goes wrong
            logg(f"writing spacy vocab to disk: {VOC_DIR}")
            # self.nlp.to_disk(SPACY_PATH)
            makedirs(VOC_DIR, exist_ok=True)
            self.nlp.vocab.to_disk(VOC_DIR)

        t1 = int(time() - t0)
        logg(f"{corpus_name}: done in {t1//3600:02d}:{(t1//60) % 60:02d}:{t1 % 60:02d}")

    def check_docs(self, text_df):
        for i, kv in enumerate(text_df.itertuples()):
            key, title, descr, text = kv
            doc = self.nlp('\n'.join(filter(None, [title, descr, text])))
            for token in doc:
                print(token.i, token.is_sent_start, token.text)

    def process_docs(self, text_df):
        """Processes DataFrames from the ETL pipeline with the NLP pipeline."""

        # process each doc in corpus
        for i, kv in tqdm(enumerate(text_df.itertuples()), total=len(text_df)):
            key, title, descr, text = kv
            # build spacy doc
            doc = self.nlp('\n'.join(filter(None, [title, descr, text])))

            # annotated phrases
            noun_chunks = {
                token.i: idx for idx, chunk in enumerate(doc.noun_chunks) for token in chunk
            }

            # extract relevant attributes
            attr = [
                {
                    HASH: key,
                    TEXT: str(token.text),
                    LEMMA: str(token.lemma_),
                    IWNLP: token._.iwnlp_lemmas,
                    POS: str(token.pos_),
                    TOK_IDX: int(token.i),
                    SENT_START: 1 if (token.is_sent_start or token.i == 0) else 0,
                    ENT_IOB: token.ent_iob_,
                    ENT_TYPE: token.ent_type_,
                    NOUN_PHRASE: noun_chunks.get(token.i, 0),
                } for token in doc
            ]

            yield self.df_from_doc(attr)

    def read(self, f, start=0, stop=None):
        """Reads a dataframe from pickle format."""

        df = pd.read_pickle(f)[[TITLE, DESCRIPTION, TEXT]].iloc[start:stop]
        # lazy hack for dewiki_new
        if 'dewiki' in f:
            good_ids = pd.read_pickle(join(ETL_DIR, 'dewiki_good_ids.pickle'))
            df = df[df.index.isin(good_ids.index)]
        self.logg(f"using {len(df):d} documents")

        return df

    def df_from_doc(self, doc):
        """
        Creates a DataFrame from a given spacy.doc that contains only nouns and noun phrases.

        :param doc: list of tokens (tuples with attributes) from spacy.doc
        :return:    pandas.DataFrame
        """

        df = pd.DataFrame.from_records(doc)
        df[ENT_IOB] = df[ENT_IOB].astype('category')
        df[ENT_TYPE] = df[ENT_TYPE].astype('category')

        # create Tokens from IWNLP lemmatization, else from spacy lemmatization (or original text)
        mask_iwnlp = ~df[IWNLP].isnull()
        df.loc[mask_iwnlp, TOKEN] = df.loc[mask_iwnlp, IWNLP]
        df.loc[~mask_iwnlp, TOKEN] = df.loc[~mask_iwnlp, LEMMA]

        # fixes wrong POS tagging for punctuation
        mask_punct = df[TOKEN].isin(list('[]<>/–%'))
        df.loc[mask_punct, POS] = PUNCT
        df[POS] = df[POS].astype('category')

        # set an index for each sentence
        df[SENT_IDX] = df[SENT_START].cumsum()

        # set an index for each entity
        df[ENT_IDX] = (df[ENT_IOB] == 'B')
        df[ENT_IDX] = df[ENT_IDX].cumsum()
        df.loc[df[ENT_IOB] == 'O', ENT_IDX] = 0

        # fix whitespace tokens
        df[TEXT] = df[TEXT].str.replace('\n', '<newline>')
        df[TEXT] = df[TEXT].str.replace('\t', '<tab>')

        df = df[self.FIELDS]

        return df
