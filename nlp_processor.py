# -*- coding: utf-8 -*-

from os import makedirs
from os.path import exists, join
from time import time
import spacy
import pandas as pd

from constants import VOCAB_PATH, TEXT, LEMMA, IWNLP, POS, INDEX, SENT_START, ENT_IOB, TOKEN, SENT_IDX, HASH, \
    NOUN_PHRASE, NLP_PATH, PUNCT, SPACE, NUM, DET, tprint
from lemmatizer_plus import LemmatizerPlus
from project_logging import log


class NLPProcessor(object):
    def __init__(self, spacy_path, lemmatizer_path='../data/IWNLP.Lemmatizer_20170501.json'):
        ### --- load spacy and iwnlp ---
        log("loading spacy")
        self.nlp = spacy.load(spacy_path)  # <-- load with dependency parser (slower)
        # nlp = spacy.load(de, disable=['parser'])

        if exists(VOCAB_PATH):
            log("reading vocab from " + VOCAB_PATH)
            self.nlp.vocab.from_disk(VOCAB_PATH)

        log("loading IWNLPWrapper")
        self.lemmatizer = LemmatizerPlus(lemmatizer_path=lemmatizer_path)
        self.nlp.add_pipe(self.lemmatizer)

    def read_process_store(self, file_path, corpus_name, store=True, vocab_to_disk=True, size=None):
        log("*** start new corpus: " + corpus_name)
        t0 = time()

        # read
        log(corpus_name + ": reading corpus from " + file_path)
        df = self.read(file_path)

        # process
        log(corpus_name + ": start processing:")
        df = [doc for doc in self.process_docs(df[TEXT], size=size)]
        df = pd.concat(df).reset_index(drop=True)

        # store
        if store:
            self.store(corpus_name, df, suffix='_nlp')
        if vocab_to_disk:
            # stored with each corpus, in case anythings goes wrong
            log("writing spacy vocab to disk: " + VOCAB_PATH)
            # self.nlp.to_disk(SPACY_PATH)
            makedirs(VOCAB_PATH, exist_ok=True)
            self.nlp.vocab.to_disk(VOCAB_PATH)

        t1 = int(time() - t0)
        log("{:s}: done in {:02d}:{:02d}:{:02d}".format(corpus_name, t1//3600, t1//60, t1 % 60))

        return df

    def process_docs(self, text_series, size=None, steps=100):
        """ main function for sending the dataframes from the ETL pipeline to the NLP pipeline """
        step_len = 100//steps
        percent = len(text_series)//steps
        done = 0
        for i, kv in enumerate(text_series[:size].iteritems()):
            # log progress
            if percent > 0 and i % percent == 0:
                log("  {:d}%: {:d} documents processed".format(done, i))
                done += step_len

            k, v = kv
            # build spacy doc
            doc = self.nlp(v)
            yield self.df_from_doc(doc, key=k)

    def df_from_doc(self, doc, key):
        """
        Creates a pandas DataFrame from a given spacy.doc that contains only nouns and noun phrases.
        :param doc: spacy.doc
        :param key: hash key from document
        :return:    pandas.DataFrame
        """
        tags = [
            (
                str(token.text), str(token.lemma_), token._.iwnlp_lemmas,
                str(token.pos_),
                # token.tag_, token.is_stop,
                int(token.i),
                # token.idx,
                int(token.is_sent_start or 0),
                # token.ent_type_,
                token.ent_iob_,
                # token.ent_id_,
            ) for token in doc
        ]
        df = pd.DataFrame(tags)
        df = df.rename(columns={k: v for k, v in enumerate([
            TEXT, LEMMA, IWNLP, POS,
            # TAG, STOP,
            INDEX,
            # START,
            SENT_START,
            # ENT_TYPE,
            ENT_IOB,
            # "Dep", "Shape", "alpha", "Ent_id"  # currently not used :(
        ])})

        # create Tokens from IWNLP lemmatization, else from spacy lemmatization (or original text)
        mask_iwnlp = ~df[IWNLP].isnull()
        df.loc[mask_iwnlp, TOKEN] = df.loc[mask_iwnlp, IWNLP]
        df.loc[~mask_iwnlp, TOKEN] = df.loc[~mask_iwnlp, LEMMA]
        df[SENT_IDX] = df[SENT_START].cumsum()

        # add phrases
        df = self.annotate_phrases(df, doc)
        # add hash-key
        df[HASH] = key

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
        return fname

    @staticmethod
    def read(f):
        """ reads a dataframe from pickle format """
        return pd.read_pickle(f)
