# -*- coding: utf-8 -*-

import sys
import re
from os import listdir
from os.path import isfile, join
from time import time

from project_logging import log
from constants import *
from nlp_processor import NLPProcessor

if len(sys.argv) > 1 and sys.argv[1] == '--hpc':
    HPC = True
    DE = '/home/funkea/.local/lib/python3.4/site-packages/de_core_news_sm/de_core_news_sm-2.0.0'

### --- run ---
log("##### START #####")

t0 = time()

LOCAL_PATH = ETL_BASE
FULL_PATH = join(DATA_BASE, LOCAL_PATH)

files = sorted([f for f in listdir(FULL_PATH) if isfile(join(FULL_PATH, f))])
# filter for certain file prefixes if file_prefix is set
file_prefix = 'Onl'
files = filter(lambda x: x[:len(file_prefix)] == file_prefix if file_prefix else True, files)
processor = NLPProcessor(spacy_path=DE)

for name in files:
    corpus = re.split(r'\.|_', name)[0]
    fname = join(FULL_PATH, name)
    df = processor.read_process_load(fname, corpus, spacy_to_disk=True, size=10)

t1 = int(time() - t0)
log("finished in {:02d}:{:02d}:{:02d} minutes".format(t1//3600, t1//60, t1 % 60))

tprint(df, 50)
