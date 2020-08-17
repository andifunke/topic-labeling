# -*- coding: utf-8 -*-
import csv
from time import time

import pandas as pd
import spacy
from ftfy import fix_text
from tqdm import tqdm

from topiclabeling.preprocessing.nlp_lemmatizer_plus import LemmatizerPlus
from topiclabeling.utils.constants import (
    VOC_DIR, TEXT, LEMMA, IWNLP, POS, TOK_IDX, SENT_START, ENT_IOB, ENT_TYPE, ENT_IDX, TOKEN,
    SENT_IDX, HASH, NOUN_PHRASE, NLP_DIR, PUNCT, TITLE, ETL_DIR, DESCRIPTION, IWNLP_FILE
)
from topiclabeling.utils.logging import logg


class NLProcessor(object):

    FIELDS = [HASH, TOK_IDX, SENT_IDX, TEXT, TOKEN, POS, ENT_IOB, ENT_IDX, ENT_TYPE, NOUN_PHRASE]

    def __init__(self, spacy_path, iwnlp_path=IWNLP_FILE, lemmatization_map_file=None):

        # ------ load spacy and iwnlp ------
        logg("loading spacy")
        self.nlp = spacy.load(spacy_path)
        # nlp = spacy.load(de, disable=['parser'])   # <-- load without dependency parser (fast)

        if VOC_DIR.exists():
            logg(f"reading vocab from {VOC_DIR}")
            self.nlp.vocab.from_disk(VOC_DIR)

        logg("loading IWNLPWrapper")
        self.lemmatizer = LemmatizerPlus(
            iwnlp_path, self.nlp, lemmatization_map_file=lemmatization_map_file)
        self.nlp.add_pipe(self.lemmatizer)
        self.stringstore = self.nlp.vocab.strings

    def read_process_store(self, file_path, corpus_name, start=0, stop=None):
        """
        Reads a dataframe from the ETL pipeline, applies the NLP pipeline and writes
        the annotations to a csv file.

        :param file_path: path to the input dataframe file in pickle format.
        :param corpus_name: name/id of the processed corpus
        :param start: if not None: first processed document at index `start`.
        :param stop: if not None: last processed document at index `stop`-1.
        """

        logg(f"*** start new corpus: {corpus_name}")
        t0 = time()

        # read the etl dataframe
        slicing = f"[{start:d}:{stop:d}]" if (start or stop) else ''
        logg(f"{corpus_name}: reading corpus{slicing} from {file_path}")
        df = self.read(file_path, start=start, stop=stop)

        # start the nlp pipeline
        logg(f"{corpus_name}: start processing")
        # self.check_docs(df); return
        reader = self.process_docs(df, ignore_title=corpus_name.startswith('dewac'))

        if start or stop:
            suffix = f'_{start:d}_{stop-1:d}_nlp'
        else:
            suffix = '_nlp'

        NLP_DIR.mkdir(exist_ok=True, parents=True)
        filename = NLP_DIR / f'{corpus_name}{suffix}.csv'
        logg(f"{corpus_name}: saving to {filename}")

        with open(filename, 'w') as fp:
            header = True
            for doc in reader:
                try:
                    doc.to_csv(fp, sep='\t', quoting=csv.QUOTE_NONE, index=None, header=header)
                except csv.Error as e:
                    logg(doc)
                    raise e
                header = False

        t1 = int(time() - t0)
        logg(f"{corpus_name}: done in {t1 // 3600:02d}:{(t1 // 60) % 60:02d}:{t1 % 60:02d}")

    def process_docs(self, text_df, fix_encoding_errors=True, ignore_title=False):
        """Processes DataFrames from the ETL pipeline with the NLP pipeline."""

        # process each doc in corpus
        for i, kv in tqdm(enumerate(text_df.itertuples()), total=len(text_df)):
            key, title, descr, text = kv
            # build spacy doc
            texts = [descr, text] if ignore_title else [title, descr, text]
            raw_text = '\n'.join(filter(None, texts))
            if fix_encoding_errors:
                raw_text = raw_text.replace('\x00', ' ')
                raw_text = fix_text(raw_text, fix_entities=True)
            doc = self.nlp(raw_text)

            # annotated phrases
            noun_chunks = {
                token.i: idx for idx, chunk in enumerate(doc.noun_chunks) for token in chunk
            }

            # extract relevant attributes
            attr = [
                {
                    HASH: key,
                    TEXT: token.text,
                    LEMMA: token.lemma_,
                    IWNLP: token._.iwnlp_lemmas,
                    POS: token.pos_,
                    TOK_IDX: token.i,
                    SENT_START: token.is_sent_start or token.i == 0,
                    ENT_IOB: token.ent_iob_,
                    ENT_TYPE: token.ent_type_,
                    NOUN_PHRASE: noun_chunks.get(token.i, -1),
                } for token in doc
            ]

            yield self.df_from_doc(attr)

    @staticmethod
    def read(f, start=0, stop=None):
        """Reads a dataframe from pickle format."""

        df = pd.read_pickle(f)[[TITLE, DESCRIPTION, TEXT]].iloc[start:stop]
        # lazy hack for dewiki_new
        if 'dewiki' in f.name:
            good_ids = pd.read_pickle(ETL_DIR / 'dewiki_good_ids.pickle')
            df = df[df.index.isin(good_ids.index)]
        logg(f"using {len(df):d} documents")

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
        df[SENT_IDX] = df[SENT_START].cumsum() - 1

        # set an index for each entity
        df[ENT_IDX] = (df[ENT_IOB] == 'B')
        df[ENT_IDX] = df[ENT_IDX].cumsum() - 1
        df.loc[df[ENT_IOB] == 'O', ENT_IDX] = -1

        # fix whitespace tokens
        df[TEXT] = df[TEXT].str.replace(' *\n *', '<newline>')
        df[TEXT] = df[TEXT].str.replace(' *\t *', '<tab>')
        df[TEXT] = df[TEXT].str.replace(' +', '<space>')
        df[TOKEN] = df[TOKEN].str.replace(' +', '<space>')

        df = df[self.FIELDS]

        return df
