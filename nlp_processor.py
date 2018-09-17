# -*- coding: utf-8 -*-

from os import makedirs
from os.path import exists, join
from time import time
import spacy
import pandas as pd
import gc

from constants import VOC_PATH, TEXT, LEMMA, IWNLP, POS, INDEX, SENT_START, ENT_IOB, TOKEN, SENT_IDX, HASH, \
    NOUN_PHRASE, NLP_PATH, PUNCT, SPACE, NUM, DET, TITLE, DESCR
from lemmatizer_plus import LemmatizerPlus
from project_logging import log
from utils import tprint

FIELDS = [HASH, INDEX, SENT_IDX, TEXT, TOKEN, POS, ENT_IOB, NOUN_PHRASE]


class NLPProcessor(object):
    def __init__(self, spacy_path, lemmatizer_path='../data/IWNLP.Lemmatizer_20170501.json'):
        ### --- load spacy and iwnlp ---
        log("loading spacy")
        self.nlp = spacy.load(spacy_path)  # <-- load with dependency parser (slower)
        # nlp = spacy.load(de, disable=['parser'])

        if exists(VOC_PATH):
            log("reading vocab from " + VOC_PATH)
            self.nlp.vocab.from_disk(VOC_PATH)

        log("loading IWNLPWrapper")
        self.lemmatizer = LemmatizerPlus(lemmatizer_path=lemmatizer_path)
        self.nlp.add_pipe(self.lemmatizer)

    def read_process_store(self, file_path, corpus_name, store=True, vocab_to_disk=True, size=None):
        log("*** start new corpus: " + corpus_name)
        t0 = time()

        # read the etl dataframe
        log(corpus_name + ": reading corpus from " + file_path)
        df = self.read(file_path)

        # start the nlp pipeline
        log(corpus_name + ": start processing:")
        df = self.process_docs(df, size=size)

        # store
        if store:
            gc.collect()
            self.store(corpus_name, df, suffix='_nlp')
        if vocab_to_disk:
            # stored with each corpus, in case anythings goes wrong
            log("writing spacy vocab to disk: " + VOC_PATH)
            # self.nlp.to_disk(SPACY_PATH)
            makedirs(VOC_PATH, exist_ok=True)
            self.nlp.vocab.to_disk(VOC_PATH)

        t1 = int(time() - t0)
        log("{:s}: done in {:02d}:{:02d}:{:02d}".format(corpus_name, t1//3600, (t1//60) % 60, t1 % 60))

        return df

    def process_docs(self, text_df, size=None, steps=100):
        """ main function for sending the dataframes from the ETL pipeline to the NLP pipeline """
        step_len = 100//steps
        percent = len(text_df) // steps
        done = 0
        docs = []
        for i, kv in enumerate(text_df[:size].itertuples()):
            # log progress
            if percent > 0 and i % percent == 0:
                log("  {:d}%: {:d} documents processed".format(done, i))
                done += step_len

            key, title, descr, text = kv
            # build spacy doc
            doc = self.nlp('\n'.join(filter(None, [title, descr, text])))

            noun_phrases = dict()
            for chunk_idx, chunk in enumerate(doc.noun_chunks, 1):
                for token in chunk:
                    noun_phrases[token.i] = chunk_idx
            tags = [[key,
                     str(token.text), str(token.lemma_), token._.iwnlp_lemmas, str(token.pos_),
                     int(token.i), int(token.is_sent_start or 0), token.ent_iob_,
                     noun_phrases.get(token.i, 0)
                     ] for token in doc]
            docs += tags

        df = self.df_from_docs(docs)
        return df

    @staticmethod
    def df_from_docs(docs):
        """
        Creates a pandas DataFrame from a given spacy.doc that contains only nouns and noun phrases.
        :param docs: list of tokens (tuples with attributes) from spacy.doc
        :return:    pandas.DataFrame
        """
        df = pd.DataFrame(docs,
                          columns=[HASH, TEXT, LEMMA, IWNLP, POS, INDEX, SENT_START, ENT_IOB, NOUN_PHRASE])
        # create Tokens from IWNLP lemmatization, else from spacy lemmatization (or original text)
        mask_iwnlp = ~df[IWNLP].isnull()
        df.loc[mask_iwnlp, TOKEN] = df.loc[mask_iwnlp, IWNLP]
        df.loc[~mask_iwnlp, TOKEN] = df.loc[~mask_iwnlp, LEMMA]
        df[SENT_IDX] = df[SENT_START].cumsum()
        return df[[HASH, INDEX, SENT_IDX, TEXT, TOKEN, POS, ENT_IOB, NOUN_PHRASE]]

    @staticmethod
    def annotate_phrases(df, doc):
        """
            given a doc process and return the contained noun phrases.
            This function is based on spacy's noun chunk detection.
        """
        df[NOUN_PHRASE] = 0

        # clean the noun chunks from spacy first
        noun_index = 0
        for chunk in doc.noun_chunks:
            if len(chunk) > 5:
                continue
            chunk_ids = set()
            for token in chunk:
                # exclude punctuation and spaces
                if token.pos_ in {PUNCT, SPACE, NUM}:
                    continue
                # exclude leading determiners
                if len(chunk_ids) == 0 and (token.pos_ == DET or token.is_stop):
                    continue
                chunk_ids.add(token.i)
                # annotate the tokens with chunk id
            if len(chunk_ids) > 1:
                noun_index += 1
                df.loc[df[INDEX].isin(chunk_ids), NOUN_PHRASE] = noun_index

        return df

    @staticmethod
    def store(corpus, df, suffix=''):
        """returns the file path where the dataframe was stores"""
        makedirs(NLP_PATH, exist_ok=True)
        fname = join(NLP_PATH, corpus + suffix + '.pickle')
        log(corpus + ': saving to ' + fname)
        df.to_pickle(fname)

    @staticmethod
    def read(f):
        """ reads a dataframe from pickle format """
        return pd.read_pickle(f)[[TITLE, DESCR, TEXT]]
