# -*- coding: utf-8 -*-

from spacy.tokens import Token
from iwnlp.iwnlp_wrapper import IWNLPWrapper
from constants import ADJ, ADV, INTJ, NOUN, PROPN, VERB, ADP, AUX, CCONJ, CONJ, DET, NUM, \
    PART, PRON, SCONJ, PUNCT, SYM, X, SPACE, PHRASE


class LemmatizerPlus(object):
    def __init__(self, lemmatizer_path):
        self.lemmatizer = IWNLPWrapper(lemmatizer_path=lemmatizer_path)
        Token.set_extension('iwnlp_lemmas', getter=self.lemmatize, force=True)
        self.lookup = {
            ('fast', ADV): 'fast',
        }

    def __call__(self, doc):
        for token in doc:
            token._.iwnlp_lemmas = self.lemmatize(token)
        return doc

    def get_lemmas(self, token):
        if self.use_plain_lemmatization:
            return self.lemmatizer.lemmatize_plain(token.text, ignore_case=self.ignore_case)
        else:
            return self.lemmatizer.lemmatize(token.text, pos_universal_google=token.pos_)

    def lemmatize(self, token):
        """
        TODO: This doc is slightly outdated
        This function uses the IWNLP lemmatizer with a few enhancements for compund nouns and nouns
        with uncommon capitalization. Can also be used to lemmatize tokens with different POS-tags.
        Do not use this function to lemmatize phrases.
        :param token: white space stripped single token (str)
        :param pos:   string constant, one of Universal tagset.
        :return: str # TODO: tuple of type (str, bool)
               value[0]: The lemma of the token if a lemma can be derived, else None.
               # TODO: value[1]: True if the token can be retrieved from the Wiktionary database as is, else False.
        """
        text = token.text.strip()
        pos = token.pos_

        if pos in {PHRASE, PUNCT, SPACE, SYM}:
            return text
        if pos in {DET, NUM}:
            return None

        if pos == PROPN:
            pos = NOUN

        if (text, pos) in self.lookup:
            return self.lookup[(text, pos)]

        value = None
        lemm = self.lemmatizer.lemmatize(text, pos)
        # default lemmatization ok?
        if lemm:
            value = lemm[0]

        # some rules to derive a lemma from the original token (nouns only)
        # TODO: define rules for hyphenated nouns
        elif pos == NOUN:
            # first try default noun capitalization
            lemm = self.lemmatizer.lemmatize(text.title(), pos)
            if lemm:
                value = lemm[0]
            else:
                # still no results: try noun suffixes
                # TODO: search for a more efficient implementation
                text_low = text.lower()
                for i in range(1, len(text) - 1):
                    text_edit = text_low[i].upper() + text_low[i+1:]
                    lemm = self.lemmatizer.lemmatize(text_edit, pos)
                    if lemm:
                        value = text[:i].title() + lemm[0].lower()
                        break

        # last try: plain lemmatization
        else:
            lemm = self.lemmatizer.lemmatize_plain(text, ignore_case=True)
            if lemm:
                value = lemm[0]

        if value and pos in {ADJ, ADP, ADV, AUX, CCONJ, CONJ, INTJ, PART, PRON, SCONJ, VERB}:
            value = value.lower()

        self.lookup[(text, pos)] = value
        return value
