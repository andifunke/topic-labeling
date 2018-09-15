from argument_parser import get_options

OPTIONS = get_options()
NOTEBOOK = False
HPC = OPTIONS['hpc']
LOG = OPTIONS['log']
LOG_PATH = OPTIONS['log_path']
CORPUS_PREFIXES = OPTIONS['corpus_prefix']
DE = OPTIONS['spacy_model_path']
