
# coding: utf-8

# In[1]:


# coding: utf-8
from os import listdir, makedirs
from os.path import join, isfile, isdir, exists
import pandas as pd
import gc
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import CoherenceModel, TfidfModel, LdaModel, LdaMulticore
from gensim.models.coherencemodel import COHERENCE_MEASURES
from gensim.models.hdpmodel import HdpModel, HdpTopicFormatter
from gensim.models.callbacks import CoherenceMetric, DiffMetric, PerplexityMetric, ConvergenceMetric
from gensim.utils import revdict
from itertools import chain, islice
from constants import (
    FULL_PATH, ETL_PATH, NLP_PATH, SMPL_PATH, POS, NOUN, PROPN, TOKEN, HASH, SENT_IDX, PUNCT
)
import logging
import json
import numpy as np

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
pd.options.display.max_rows = 2001

from gensim import utils



#report_on_oov_terms(cm, trained_models.values())

datasets = {
    'E': 'Europarl',
    'FA': 'FAZ_combined',
    'FO': 'FOCUS_cleansed',
    'O': 'OnlineParticipation',
    'P': 'PoliticalSpeeches',
    'dewi': 'dewiki',
    'dewa': 'dewac',
}
bad_tokens = {
    'Europarl': [
        'E.', 'Kerr', 'The', 'la', 'ia', 'For', 'Ieke', 'the',
    ],
    'FAZ_combined': [
        'S.', 'j.reinecke@faz.de', 'B.',
    ],
    'FOCUS_cleansed': [],
    'OnlineParticipation': [
        'Re', '@#1', '@#2', '@#3', '@#4', '@#5', '@#6', '@#7', '@#8', '@#9', '@#1.1', 'Für', 'Muss',
        'etc', 'sorry', 'Ggf', 'u.a.', 'z.B.', 'B.', 'stimmt', ';-)', 'lieber', 'o.', 'Ja', 'Desweiteren',
    ],
    'PoliticalSpeeches': [],
    'dewiki': [],
    'dewac': [],
}
all_bad_tokens = set(chain(*bad_tokens.values()))
params_list = ['a42', 'b42', 'c42', 'd42', 'e42']
placeholder = '[[PLACEHOLDER]]'


# In[2]:


dataset = datasets['O']
file = f'{dataset}_LDAmodel_a42_10'
nbtopics = int(file.split('_')[-1])
topn = 20

ldamodel = LdaModel.load(join(ETL_PATH, 'LDAmodel', 'a42', file))
dict_from_model = ldamodel.id2word
dict_from_model.add_documents([[placeholder]])
print(dict_from_model)
print(dict_from_model.id2token[0])
dfm = list(dict_from_model.id2token.values())


# In[5]:


split_type = 'fullset'
corpus_type = 'bow'
file_name = f'{dataset}_{split_type}_nouns_{corpus_type}'
corpus_path = join(ETL_PATH, 'LDAmodel', file_name + '.mm')
dict_path = join(ETL_PATH, 'LDAmodel', file_name + '.dict')
dict_from_corpus = Dictionary.load(dict_path)
dict_from_corpus.add_documents([[placeholder]])
dict_from_corpus[0]
print(len(dict_from_corpus.token2id), len(dict_from_corpus.id2token))
dfc = list(dict_from_corpus.id2token.values())


# In[6]:


set(dfm).difference(set(dfc))


# In[10]:


corpus = MmCorpus(corpus_path)
print(corpus)


# In[36]:


list(corpus)


# In[12]:


topics = [
    [dataset] +
    [dict_from_model[term[0]] for term in ldamodel.get_topic_terms(i, topn=topn)]
    for i in range(nbtopics)
]
df_topics = pd.DataFrame(topics, columns=['dataset'] + ['term' + str(i) for i in range(topn)])
df_topics = df_topics.applymap(lambda x: placeholder if x in all_bad_tokens else x)
df_topicsIds = df_topics.iloc[:, 1:].applymap(lambda x: dict_from_corpus.token2id[x])
topics = df_topics.iloc[:, 1:].values.tolist()
topicsIds = df_topicsIds.values.tolist()

#cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=dict_from_corpus, coherence='u_mass', topn=topn)
#cm.get_coherence_per_topic(
    #segmented_topics=topics, with_std=False, with_support=False
#)

doc_path = join(ETL_PATH, 'LDAmodel', file_name.rstrip(f'{corpus_type}') + 'texts.json')
with open(doc_path, 'r') as fp:
    texts = json.load(fp)

# this!
texts.append([placeholder])

cm = CoherenceModel(topics=topics, texts=texts, dictionary=dict_from_corpus, coherence='c_v', topn=topn)
x = cm.get_coherence_per_topic(
    #segmented_topics=topics, with_std=False, with_support=False
)
print(x)


umass_segmented_topics = COHERENCE_MEASURES['c_v'].seg(topicsIds)
print(umass_segmented_topics)
#print(cm.segment_topics())

#from pprint import pprint
#pprint(topics)

x = cm.get_coherence_per_topic(
    segmented_topics=umass_segmented_topics,
    # with_std=False, with_support=False
)
print(x)
