# -*- coding: utf-8 -*-

from os import makedirs
from os.path import exists
import pandas as pd
import spacy

from lemmatizer_plus import LemmatizerPlus
from constants import *
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

    def read_process_load(self, file_path, corpus_name, spacy_to_disk=False, size=None):
        df = self.read(file_path)
        log("processing " + corpus_name)
        # docs, phrase_lookups = zip(*[tple for tple in process_docs(df[TEXT], size=None)])
        docs = [doc for doc in self.process_docs(df[TEXT], size=size)]
        docs = pd.concat(docs).reset_index(drop=True)
        self.store(corpus_name + '_nlp', docs)
        # phrase_lookups = pd.concat(phrase_lookups).reset_index(drop=True)
        # store(corpus_name + '_phrase_lookups', phrase_lookups)
        if spacy_to_disk:
            # stored with each corpus, in case anythings goes wrong
            log("writing spacy model to disk: " + NLP_PATH)
            self.nlp.to_disk(SPACY_PATH)
        # self.nlp.vocab.to_disk(VOCAB_PATH)
        return docs

    def process_docs(self, series, size=None):
        """ main function for sending the dataframes from the ETL pipeline to the NLP pipeline """
        length = len(series)
        steps = 100
        step_len = 100 // steps
        percent = length // steps
        done = 0
        for i, kv in enumerate(series[:size].iteritems()):
            if percent > 0 and i % percent == 0:
                log("{:d}%: {:d} documents processed".format(done, i))
                done += step_len

            k, v = kv
            # build spacy doc
            doc = self.nlp(v)
            # TODO: version with phrase detection and phrase lookup table
            # essential_token, phrase_lookup = essence_from_doc(doc, key=k)
            # yield essential_token, phrase_lookup
            yield self.df_from_doc(doc, key=k)

    @staticmethod
    def process_phrases(doc):
        """
            given a doc process and return the contained noun phrases.
            This function is based on spacy's noun chunk detection.
            It also creates items for a global phrase lookup table, which are currently not used.
        """
        # clean the noun chuncs from spacy first
        noun_chunks = []
        for chunk in doc.noun_chunks:
            if len(chunk) > 5:
                continue
            start = False
            noun_chunk = []
            for token in chunk:
                # exclude punctuation
                if token.pos_ == PUNCT:
                    continue
                # exclude leading determiners
                if not start and (token.pos_ == DET or token.is_stop):
                    continue
                start = True
                noun_chunk.append(token)
            if len(noun_chunk) > 1:
                noun_chunks.append(noun_chunk)

        # the remaining, adjusted noun chunks will be lemmatized and indexed
        phrase_list_lookup = []
        phrase_list_doc = []
        for chunk in noun_chunks:
            phrase = []
            text = []
            for token in chunk:
                lemma = token._.iwnlp_lemmas
                if not lemma:
                    lemma = token.lemma_
                phrase.append(lemma)
                text.append(token.text)
            phrase = ' '.join(phrase)
            text = ' '.join(text)

            # add to phrase collection of corpus
            phrase_lookup = pd.Series()
            phrase_lookup['lemmatized'] = phrase
            phrase_lookup['original'] = text
            # phrase_lookup['Spacy Tokens'] = tuple(chunk)
            phrase_list_lookup.append(phrase_lookup)

            # add to document dataframe
            phrase_dic = dict()
            phrase_dic[TEXT] = text
            phrase_dic[TOKEN] = phrase
            phrase_dic[POS] = PHRASE
            phrase_dic[INDEX] = chunk[0].i
            # phrase_dic[START] = chunk[0].idx
            phrase_dic[SENT_START] = 0
            phrase_list_doc.append(phrase_dic)

        # return the dataframes and for the doc dataframe and for the global phrase lookup table
        return pd.DataFrame(phrase_list_doc), pd.DataFrame(phrase_list_lookup)

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
                # token.ent_type_, token.ent_iob_,  # token.ent_id_,
            ) for token in doc
        ]
        df = pd.DataFrame(tags)
        df = df.rename(columns={k: v for k, v in enumerate([
            TEXT, LEMMA, IWNLP, POS,
            # TAG, STOP,
            INDEX,
            # START,
            SENT_START,
            # ENT_TYPE, ENT_IOB,
            # "Dep", "Shape", "alpha", "Ent_id"  # currently not used :(
        ])})

        # create Tokens from IWNLP lemmatization, else from spacy lemmatization (or original text)
        mask_iwnlp = ~df[IWNLP].isnull()
        df.loc[mask_iwnlp, TOKEN] = df.loc[mask_iwnlp, IWNLP]
        df.loc[~mask_iwnlp, TOKEN] = df.loc[~mask_iwnlp, LEMMA]

        # add phrases
        df_phrases, phrase_lookup = self.process_phrases(doc)
        df = df.append(df_phrases).sort_values(INDEX)

        df[SENT_IDX] = df[SENT_START].cumsum()

        # add hash-key
        df[HASH] = key

        return df[[HASH, INDEX, SENT_IDX, TEXT, TOKEN, POS]]  #, phrase_lookup

    @staticmethod
    def store(corpus, df):
        """returns the file path where the dataframe was stores"""
        makedirs(NLP_PATH, exist_ok=True)
        fname = join(NLP_PATH, corpus + '.pickle')
        log('saving' + corpus + ' to ' + fname)
        df.to_pickle(fname)
        return fname

    @staticmethod
    def read(f):
        """ reads a dataframe from pickle format """
        log("reading corpus from " + f)
        return pd.read_pickle(f)
