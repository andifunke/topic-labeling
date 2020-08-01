# -*- coding: utf-8 -*-

from iwnlp.iwnlp_wrapper import IWNLPWrapper
from spacy.tokens import Token

from topic_labeling.utils.constants import (
    ADJ, ADV, INTJ, NOUN, PROPN, VERB, ADP, AUX, CCONJ, CONJ, DET, NUM,
    PART, PRON, SCONJ, PUNCT, SYM, SPACE, PHRASE, NPHRASE
)


class LemmatizerPlus(object):

    def __init__(self, lemmatizer_path, nlp):
        self.lemmatizer = IWNLPWrapper(lemmatizer_path=lemmatizer_path)
        self.stringstore = nlp.vocab.strings

        Token.set_extension('iwnlp_lemmas', getter=self.lemmatize, force=True)
        self.lookup = {('fast', ADV): 'fast'}

    def __call__(self, doc):
        for token in doc:
            token._.iwnlp_lemmas = self.lemmatize(token)
        return doc

    def lemmatize(self, token, tolerance=3):
        """
        TODO: This doc is slightly outdated
        This function uses the IWNLP lemmatizer with a few enhancements for compound nouns
        and nouns with uncommon capitalization. Can also be used to lemmatize tokens with
        different POS-tags. Do not use this function to lemmatize phrases.

        :param token: white space stripped single token (str)
        :param tolerance: min number of chars to infer compound lemmata.
        :return: str # TODO: tuple of type (str, bool)
               value[0]: The lemma of the token if a lemma can be derived, else None.
               # TODO: value[1]: True if the token can be retrieved from the Wiktionary
               #       database as is, else False.
        """
        text = token.text.strip()
        pos = token.pos_

        # nothing to lemmatize here
        if pos in {PHRASE, NPHRASE, PUNCT, SYM}:
            return text
        # lemmatizations are odd on DET and NUM, so better leave it alone
        if pos in {DET, NUM}:
            return None
        # custom lemmata for whitespace
        if pos == SPACE:
            text = token.text.strip(' ')
            if text == '\n':
                return '<newline>'
            if text == '\t':
                return '<tab>'
            return '<space>'

        # Wiktionary has no POS PROPN
        if pos == PROPN:
            pos = NOUN

        # first lookup token for given POS in dictionary
        if (text, pos) in self.lookup:
            return self.lookup[(text, pos)]

        value = None
        # default IWNLP lemmatization
        lemmata = self.lemmatizer.lemmatize(text, pos)
        # default lemmatization hit?
        if lemmata:
            value = text if text in lemmata else lemmata[0]

        # default lemmatization miss?
        # apply some rules to derive a lemma from the original token (nouns only)
        elif pos == NOUN:
            # first try default noun capitalization
            lemmata = self.lemmatizer.lemmatize(text.title(), pos)
            if lemmata:
                value = text if text in lemmata else lemmata[0]
            else:
                # still no results: try all noun suffixes
                # TODO: search for a more efficient implementation
                text_low = text.lower()
                for i in range(1, len(text) - tolerance):
                    # looks ugly, but avoids full capitalization
                    text_edit = text_low[i].upper() + text_low[i+1:]
                    lemmata = self.lemmatizer.lemmatize(text_edit, pos)
                    if lemmata:
                        value = text_edit if text_edit in lemmata else lemmata[0]
                        value = (text[:i] + value).title()
                        break

        # last try: plain lemmatization for all remaining POS tags
        else:
            lemmata = self.lemmatizer.lemmatize_plain(text, ignore_case=True)
            if lemmata:
                value = text if text in lemmata else lemmata[0]

        if value and pos in {ADJ, ADP, ADV, AUX, CCONJ, CONJ, INTJ, PART, PRON, SCONJ, VERB}:
            value = value.lower()

        if value:
            self.stringstore.add(value)
            self.lookup[(text, pos)] = value
        return value
