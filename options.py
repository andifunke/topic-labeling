""" parsing arguments and setting default values """
import argparse


NOTEBOOK = False
HPC = False
LOG = True
STORE = True
LOG_PATH = './../logs/nlp_pipe.log'
CORPUS_PREFIXES = ''
DE = 'de'
DE_HPC = '/home/funkea/.local/lib/python3.4/site-packages/de_core_news_sm/de_core_news_sm-2.0.0'


def update_from_args():
    global HPC, LOG, STORE, LOG_PATH, CORPUS_PREFIXES, DE

    parser = argparse.ArgumentParser(description='topic labeling project')

    # boolean
    parser.add_argument('--hpc', dest='hpc', action='store_true', required=False)
    parser.add_argument('--no-hpc', dest='hpc', action='store_false', required=False)
    parser.set_defaults(hpc=HPC)
    parser.add_argument('--log', dest='log', action='store_true', required=False)
    parser.add_argument('--no-log', dest='log', action='store_false', required=False)
    parser.set_defaults(log=LOG)
    parser.add_argument('--store', dest='store', action='store_true', required=False)
    parser.add_argument('--no-store', dest='store', action='store_false', required=False)
    parser.set_defaults(store=STORE)

    # strings / paths
    parser.add_argument('--spacy_model_path', type=str, required=False)
    parser.add_argument('--log_path', type=str, required=False)
    parser.add_argument('--corpus_prefix', nargs='*', required=False)

    options = vars(parser.parse_args())

    HPC = options['hpc']
    LOG = options['log']
    STORE = options['store']
    if options['log_path']:
        LOG_PATH = options['log_path']
    if options['corpus_prefix']:
        CORPUS_PREFIXES = options['corpus_prefix']
    if options['spacy_model_path']:
        DE = options['spacy_model_path']
    if options['hpc']:
        DE = DE_HPC

    return options
