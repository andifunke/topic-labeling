import spacy
import logging


def extract_german_groups(tokens):
    token_index = 0
    new_tokens = []
    ner_filter = ['PERSON', 'LOC']
    ner_inner_labels = ['I', 'L']
    while token_index < len(tokens):
        token = tokens[token_index]
        if token.ent_type_ in ner_filter:
            ner_group = [token.text]
            inner_index = token_index + 1
            while inner_index < len(tokens) and \
                            tokens[inner_index].ent_type_ == token.ent_type_ and \
                            tokens[inner_index].ent_iob_ in ner_inner_labels:
                ner_group.append(tokens[inner_index].text)
                inner_index += 1
                token_index += 1
            new_tokens.append('_'.join(ner_group))
        else:
            new_tokens.append(token.text)
        token_index += 1
    return new_tokens


class SpacyNERGrouper(object):
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug('Loading Spacy model')
        self.nlp = spacy.load('de')
        self.logger.debug('Spacy model loaded')

    def group_texts(self, text):
        input_text = text.replace('-', '_')
        return extract_german_groups([token for token in self.nlp(input_text)])
