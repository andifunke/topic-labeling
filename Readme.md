| Last modified: *2019-01-09* |
|-------------:|

<img align="right" width="300" height="200" src="img/wordloud2.png">

# Topic Labeling

Topic Labeling is the process of finding or generating appropriate labels to document topics 
which were derived from the multinomial topic distributions over words inferred from a topic model
architecture such as *Latent Dirichlet Allocation* (LDA). Topics are usually represented by their top-10 
terms, i.e. by the words with the highest probability in a topic distribution. Topic labels should be
coherent with these topic terms and represent the common theme that these terms – hopefully – share.

This project proposes a framework to apply topic models on a text-corpus and eventually topic labels
on the generated topics. The framework and its NLP-pipeline is focused on corpora from the German 
language, but can be adjusted to any language of your choice. The Labeling methodology is based on an
approach by [Bhatia et.al.](http://www.aclweb.org/anthology/C16-1091) using neural embedded 
Wikipedia-titles as label candidates and ranking the most similar candidates with a SVR-ranker model.
We have transferred this approach to the Python 3 world and gently adapted the code to out framework.

However, this project goes beyond simple topic labeling. We will also present code to rerank the top-10 
topic terms in order to provide a more coherent representation before trying to find a matching label.
We will also undergo a thorough evaluation of topic coherence metrics based on ratings from human judges
for word similarities and relatedness. Additionally, we will present a new coherence measure based on 
neural word embeddings which is more efficient compared to PMI-based coherence metrics and correlates 
at least equally well with human judgments.

This project is rooted in my master's thesis on ***Topic Labeling***. During my research we generated two
annotated datasets for **a)** measuring topic model quality and evaluating topic reranking methods and 
**b)** generating a gold-standard for topic labeling for the German language.


#### Requirements

Python ≥ 3.6 is required. Other main requirements are [pandas](https://pandas.pydata.org/) 0.23.4, 
[spaCy](https://spacy.io/) 2.0.11 and [gensim](https://radimrehurek.com/gensim/) 3.5 
(with some additional proprietary fixes on which we will expand later). Please refer to 
[Requirements.txt](Requirements.txt) for details on the required python modules.

You will also need to download text-corpora, a few additional data files and external tools, 
but we will guide you step-by-step throughout this process.

---

## Processing steps and pipelines:

The data processing and evaluation is split into the following steps and pipelines. 
Pipelines persist their results to the file system and make them available for the succeeding pipeline.
We will collect example text-corpora from the web, transform them into structured/annotated data,
train LDA-topic models and word2vec-language-models, evaluate and rerank the top-10 topic terms, generate
label candidates and select the best labels. Additionally we will evaluate topic coherence metrics in 
order to make informed decisions on the topic quality.


### 1. Data gathering:

In this project we are using 10 text-corpora in total which were derived from 7 different sources. These
are:

1) **dewac**
2) **dewac1** *(subset of dewac)*
3) **dewiki**
4) **Europarl**
5) **German Political Speeches**
6) **Speeches** *(combining Europarl and German Political Speeches)*
6) **FAZ**
7) **Focus**
9) **News** *(combining FAZ and Focus)*
10) **Online Participation**

#### 1.1 Collecting freely available and edited datasets

The following corpora can be downloaded for free. Dewac requires registration and is only available
for academia.

  - [dewac](http://wacky.sslmit.unibo.it/doku.php?id=download): dewac_preproc.gz
  - [dewiki](https://dumps.wikimedia.org/dewiki/latest/): dewiki-latest-pages-articles.xml.bz2
  - [Europarl](http://www.statmt.org/europarl/): europarl.tgz
  - [German Political Speeches](http://adrien.barbaresi.eu/corpora/speeches/): 
    German-political-speeches-2018-release.zip

#### 1.2 Crawling datasets from the web

  - [faz/focus](src/scrapy/readme.txt): use the provided [scrapy](https://scrapy.org/) -spiders to fetch 
    news-articles from the web.
  - [OnlineParticipation](https://github.com/Liebeck/OnlineParticipationDatasets): Follow the 
    instructions in the linked repository to download this collection.

After download, the datasets are assumed to reside inside [data/corpora/](data/corpora). Please refer to 
[data/readme.txt](data/readme.txt) for detailed instructions on how to arrange the files. 
If you prefer a different file tree you might want to change to respective paths inside 
import_datasets.py.


---

***! The following instructions are currently subject to revision ...***

1) **Preprocessing:**
    - ETL-pipeline &rarr; NLP-pipeline &rarr; Phrase-extraction (aka 'simple') pipeline &rarr; MM-corpus

2) **Training:**
    - d2v / w2v / fastText training
    - LDA training &rarr; topic reranking &rarr; label generation
    
3) **Annotation:**
    - rating labels by human judges 

4) **Label ranking**

5) **Evaluation**
    - LDA inference
    - topic reranking metrics
    - labeling methods


#### Universal usage:

###### Information

- [overview_project.ipynb](overview_project.ipynb):<br>
an early overview of the project. This notebook is fairly outdated, though.

- [overview_stats.ipynb](overview_stats.ipynb):<br>
statistics about the datasets. Somewhat outdated.

- [overview_dewiki.ipynb](overview_dewiki.ipynb):<br>
an overview over the different corpus-files and stages of processing for the **dewiki** dataset. 
Is also informative for other datasets. The **dewiki** dataset is however usually more complex than
other datasets.

- [overview_netl.ipynb](overview_netl.ipynb):<br>
provides an overview over the labeling methods and gold standard datasets from Lau et.al. (2011) and
Bhatia et.al (2016). This serves just a reference.

###### Helper utilities

- [constants.py](constants.py):<br>
defines constants for the project used in most of the scripts and notebooks.

- [utils.py](utils.py):<br>
provides some helper functions for printing/logging and loading of data which are used universally.
Earlier scripts may not make full usage of the functions provided here.

- [options.py](options.py):<br>
provides argument parsing in the preprocessing pipelines. Has been replaced lately by more convenient 
methods in the **utils** module.

- [scrapy](scrapy):<br>
the crawler includes two working spiders for faz.net and focus.de. The choice for these news-sites is
merely due to alphabetical reasons.


#### Preprocessing:

###### ETL-pipeline

- [etl_pipeline.ipynb](etl_pipeline.ipynb):<br>
converts the raw datasets (e.g. crawled feeds) to the common data scheme. Adds unique document ids.
Does not contain extraction methods for Wikipedia \[&rarr; refer to etl_wikipedia.py\]. It is desirable
to replace this notebook with a standard python script.

- [etl_wikipedia_addon.ipynb](etl_wikipedia_addon.ipynb):<br>
converts results from wiki-extractor to the common data scheme.

- [etl_wikipedia.py](etl_wikipedia.py):<br>
first extraction of wikipedia. Has later been replaced with wiki extractor but still provides useful 
extraction of links and categories in Wikipedia articles.

###### NLP-pipeline

- [nlp_pipeline.py](nlp_pipeline.py):<br>
full nlp-pipeline based on the spacy framework and the German language models. Includes tokenization,
sentence splitting, POS-tagging, lemmatization (based on an extension of the IWNLP lemmatizer), named 
entity recognition and dependency parser (for noun-chunk tagging). Already fixes some common 
misclassification issues, but in general the accuracy of the spacy nlp-pipeline could be further 
improved.

- [nlp_processor.py](nlp_processor.py):<br>
main class for the nlp-pipeline.

- [nlp_lemmatizer_plus.py](nlp_lemmatizer_plus.py):<br>
offers additional features esp. for compound nouns.

- [nlp_lemmatization_map.ipynb](nlp_lemmatization_map.ipynb):<br>
creates a mapping from lemmatized to unlemmatized tokens. This is especially important for
phrases which are handled as single tokens. Lemmatized phrases tend to have grammatically incorrect
properties. Also creates a lemmatization map for Wiktionary terms. This helps to evaluate the quality
of topics.

###### PHRASE-pipeline

- [phrase_pipeline.py](phrase_pipeline.py):<br>
extracts noun phrases based on NERs and noun-chunks, a additional street heuristic and Wikipedia titles. 
Concatenates token to a single phrase. Is also known as 'simple' pipeline since it stores just a minimal
set of columns. The 'simple' terminology somewhat ugly and needs to be refactored.

[//]: # "- [Add_wikipedia_title_phrases.ipynb](Add_wikipedia_title_phrases.ipynb):<br>
fixed missing wiki_phrases in certain datasets. Adds title phrases from Wikipedia to files from the 
'simple' pipeline (after applying phrase extraction). This step has already been added to the 
phrase_extraction pipeline and is not required anymore."

[//]: # "- [add_wikipedia_title_phrases.py](add_wikipedia_title_phrases.py):<br>
same as notebook."

###### Other

- [preprocessing_corpus_generation.py](preprocessing_corpus_generation.py):<br>
converts a dataset to a bow or optional tf-idf representation and serializes it to the Matrix Market 
format. Also creates a gensim dictionary and additionally raw texts for LDA training. Specify the POS 
tags a training corpus should consist of (defaults to NOUN, PROPN, NER and NOUN PHRASE).

- [preprocessing_netl.ipynb](preprocessing_netl.ipynb):<br>
adapts the NETL preprocessing steps to the common data scheme. These adaptions are already incorporated
into the train_w2v|d2v.py scripts, making this notebook obsolete.

- [preprocessing_snippets.ipynb](preprocessing_snippets.ipynb):<br>
contains numerous small steps and fixes in the preprocessing pipeline. Has grown too large and has become
somewhat convoluted. Certainly needs an overhaul and some more refactoring into the official pipeline.
Most important feature is the detection and removal of inappropriate documents.


#### Training:

###### Embeddings

- [train_d2v.py](train_d2v.py):<br>
trains document embeddings from Wikipedia articles.

- [train_w2v.py](train_w2v.py):<br>
trains word embeddings from Wikipedia articles. Supports word2vec and fastText.

- [train_utils.py](train_utils.py):<br>
provides argument parsing and callback classes for word embedding inference.

###### Topic modeling

- [train_lda.py](train_lda.py):<br>
trains LDA models from datasets. Corpora have to be provided in Matrix Market format (bow or tf-idf) 
with an affiliated dictionary. Training can be observed via several callback metrics. For perplexity
evaluation a hold out set will be split from the corpus. For window based coherence metrics the corpus
has also be provided in plain text format.

- [train_lsi.py](train_lsi.py):<br>
additional LSI model training. Takes data in identical format to the LDA model.

###### Topic reranking

- [topic_reranking.py](topic_reranking.py):<br>
provides several metrics to choose the M most coherent topic representatives from the top N terms 
of a topic (where M < N and usually M = 10, N = 20). Improves human interpretability of a topic
and eases label generation due to a reduced number of outlier terms. Also includes evaluation metrics
for the quality of the reranking metrics.

- [topic_reranking_vectorbased.ipynb](topic_reranking_vectorbased.ipynb):<br>
new approach to rerank topic terms based on their similarity in embedding space. Needs to be included
into the topic_reranking.py module.

###### Label generation

- [labeling.py](label_generation.py):<br>
Generates topic label candidates based on their similarity to doc2vec, word2vec and fastText embeddings 
from Wikipedia articles. The algorithm is taken from the NETL framework by Bhatia et.al.. Copyright 
remarks will be added upon final release.

- [labeling_alternatives.ipynb](labeling_alternatives.ipynb):<br>
playground to experiment with alternative approaches to generate labels.

- [labeling_pygermanet.ipynb](labeling_pygermanet.ipynb):<br>
Contributes to the alternative label generation approaches.

###### Other

- [labeling_postprocessing.ipynb](labeling_postprocessing.ipynb):<br>
converts lemmatized terms and phrases in topics and labels to more human readable unlemmatized form.
Uses lemmatization maps. 


#### Evaluation:

- [eval_lda.ipynb](eval_lda.ipynb):<br>
Reads and plots statistics from the LDA training. Also tests, for which datasets and parameter 
combinations LDA models have been trained.

- [eval_lda_on_wikipedia.py](evaluate_topics.py):<br>
evaluates the topic coherence on basis of full Wikipedia dataset. Supports *U<sub>mass</sub>*, 
*C<sub>uci</sub>*/*C<sub>pmi</sub>*, *C<sub>npmi</sub>* and *C<sub>v</sub>* metrics.

- [eval_topic_reranking.ipynb](eval_topic_reranking.ipynb):<br>
Evaluates and plots the reranking results. Also first attempts to rate methods based on human scores.
