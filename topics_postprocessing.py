import json
from os.path import join

import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel

from constants import LDA_PATH, BAD_TOKENS, PLACEHOLDER, DSETS, NOUN_PATTERN
from utils import load, tprint


class Unlemmatizer(object):

    def __init__(self):
        self.phrases = load('phrases', 'minimal')
        self.wiktionary = load('wikt', 'lemmap').set_index('Lemma').query('POS == "Noun"')
        # tprint(self.wiktionary, 100)

    def unlemmatize_token(self, token, lemmap=None):
        if lemmap is not None and token in lemmap:
            word = lemmap[token].index[0]
        elif token in self.phrases.index:
            # print('token in phrases')
            words = self.phrases[token]
            if isinstance(words, str):
                word = words
            else:
                if token in words.values:
                    word = token
                else:
                    wc = words.value_counts()
                    word = wc.index[0]
        elif '_' in token:
            print(token)
            tokens = token.split('_')
            ts = []
            for t in tokens:
                print(t)
                if t in self.wiktionary:
                    print('token in wikt')
                    print(self.wiktionary.loc[t])
                    ts.append(t)
                elif t.title() in self.wiktionary:
                    print('token.lower in wikt')
                    print(self.wiktionary.loc[t.title()])
                    ts.append(t)
                else:
                    ts.append(t)
            word = '_'.join(ts)
            print(word)
        else:
            print('not in index')
            word = token
            print('   ', token, '->', word)
        word = word.replace('_.', '.').replace('_', ' ')
        return word

    def unlemmatize_group(self, group):
        lemmap = load(group.name, 'lemmap')
        return group.applymap(lambda x: self.unlemmatize_token(x, lemmap))

    def unlemmatize_topics(self, topics, dataset=None):
        topics = topics.copy()
        if dataset is not None:
            lemmap = load(dataset, 'lemmap')
            topics = topics.applymap(lambda x: self.unlemmatize_token(x, lemmap))
        else:
            topics = topics.groupby('dataset', sort=False).apply(self.unlemmatize_group)
        return topics


# --------------------------------------------------------------------------------------------------
# --- TopicLoader Class ---


class TopicsLoader(object):

    def __init__(
            self, dataset, param_ids='e42', nbs_topics=100,
            version='noun', corpus_type='bow', epochs=30, topn=20,
            filter_bad_terms=False, include_weights=False,
            include_corpus=False, include_texts=False,
    ):
        self.topn = topn
        self.dataset = DSETS.get(dataset, dataset)
        self.version = version
        self.param_ids = [param_ids] if isinstance(param_ids, str) else param_ids
        self.nb_topics_list = [nbs_topics] if isinstance(nbs_topics, int) else nbs_topics
        self.nb_topics = sum(self.nb_topics_list) * len(self.param_ids)
        self.corpus_type = corpus_type
        self.epochs = f"ep{epochs}"
        self.directory = join(LDA_PATH, self.version)
        self.data_filename = f'{self.dataset}_{version}'
        self.filter_terms = filter_bad_terms
        self.include_weights = include_weights
        self.dict_from_corpus = self._load_dict()
        self.topics = self._topn_topics()
        self.corpus = self._load_corpus() if include_corpus else None
        self.texts = self._load_texts() if include_texts else None

    def _topn_topics(self):
        """
        get the topn topics from the LDA-model in DataFrame format
        """
        all_topics = []
        for param_id in self.param_ids:
            for nb_topics in self.nb_topics_list:
                ldamodel = self._load_model(param_id, nb_topics)
                # topic building ignoring placeholder values
                topics = []
                topics_weights = []
                for i in range(nb_topics):
                    tokens = []
                    weights = []
                    for term in ldamodel.get_topic_terms(i, topn=self.topn*2):
                        token = ldamodel.id2word[term[0]]
                        weight = term[1]
                        if self.filter_terms and (token in BAD_TOKENS or NOUN_PATTERN.match(token)):
                            continue
                        else:
                            tokens.append(token)
                            weights.append(weight)
                            if len(tokens) == self.topn:
                                break
                    topics.append(tokens)
                    topics_weights.append(weights)

                model_topics = (
                    pd.DataFrame(topics, columns=[f'term{i}' for i in range(self.topn)])
                    .assign(
                        dataset=self.dataset,
                        param_id=param_id,
                        nb_topics=nb_topics
                    )
                )
                if self.include_weights:
                    model_weights = (
                        pd.DataFrame(topics_weights, columns=[f'weight{i}' for i in range(self.topn)])
                    )
                    model_topics = (
                        pd.concat([model_topics, model_weights], axis=1, sort=False)
                    )
                all_topics.append(model_topics)
        topics = (
            pd.concat(all_topics)
            .rename_axis('topic_idx')
            .reset_index(drop=False)
            .set_index(['dataset', 'param_id', 'nb_topics', 'topic_idx'])
        )
        return topics

    def topic_ids(self):
        return self.topics.applymap(lambda x: self.dict_from_corpus.token2id[x])

    def _load_model(self, param_id, nb_topics):
        """
        Load an LDA model.
        """
        model_dir = join(self.directory, self.corpus_type, param_id)
        model_file = f'{self.dataset}_LDAmodel_{param_id}_{nb_topics}_{self.epochs}'
        model_path = join(model_dir, model_file)
        print('Loading model from', model_path)
        ldamodel = LdaModel.load(model_path)
        return ldamodel

    def _load_dict(self):
        """
        This dictionary is a different from the model's dict with a different word<->id mapping,
        but from the same corpus and will be used for the Coherence Metrics.
        """
        dict_dir = join(self.directory, self.corpus_type)
        dict_path = join(dict_dir, f'{self.data_filename}_{self.corpus_type}.dict')
        print('loading dictionary from', dict_path)
        dict_from_corpus: Dictionary = Dictionary.load(dict_path)
        dict_from_corpus.add_documents([[PLACEHOLDER]])
        _ = dict_from_corpus[0]  # init dictionary
        return dict_from_corpus

    def _load_corpus(self):
        """
        load corpus (for u_mass scores)
        """
        corpus_dir = join(self.directory, self.corpus_type)
        corpus_path = join(corpus_dir, f'{self.data_filename}_{self.corpus_type}.mm')
        print('loading corpus from', corpus_path)
        corpus = MmCorpus(corpus_path)
        corpus = list(corpus)
        corpus.append([(self.dict_from_corpus.token2id[PLACEHOLDER], 1.0)])
        return corpus

    def _load_texts(self):
        """
        load texts (for c_... scores using sliding window)
        """
        doc_path = join(self.directory, self.data_filename + '_texts.json')
        with open(doc_path, 'r') as fp:
            print('loading texts from', doc_path)
            texts = json.load(fp)
        texts.append([PLACEHOLDER])
        return texts


def main():
    dataset = 'news'
    # param_ids = 'e42'

    tl = TopicsLoader(
        dataset=dataset,
        # param_ids=param_ids,
        # nbs_topics=nbs_topics,
        # version=version,
        # topn=nb_candidate_terms
    )
    # tprint(tl.topics)
    ul = Unlemmatizer()
    topics = ul.unlemmatize_topics(tl.topics)
    tprint(topics)


if __name__ == '__main__':
    main()
