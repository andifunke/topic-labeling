# -*- coding: utf-8 -*-

import re
from os import listdir
from os.path import isfile, join
from time import time

import options
options.update_from_args()
from options import CORPUS_PREFIXES, DE, STORE

from constants import FULL_PATH
from nlp_processor import NLPProcessor
from project_logging import log


if __name__ == "__main__":
    t0 = time()

    ### --- run ---
    log("##### START #####")

    # filter files for certain prefixes
    prefixes = r'^(' + '|'.join(CORPUS_PREFIXES) + r').'
    pattern = re.compile(prefixes)
    files = sorted([f for f in listdir(FULL_PATH)
                    if (isfile(join(FULL_PATH, f)) and pattern.match(f))])
    processor = NLPProcessor(spacy_path=DE)

    for name in files:
        corpus = name.split('.')[0]
        fname = join(FULL_PATH, name)
        processor.read_process_store(fname, corpus,
                                     store=STORE,
                                     vocab_to_disk=STORE,
                                     # size=10,
                                     # print=True,
                                     # head=1000,
                                     )

    t1 = int(time() - t0)
    log("all done in {:02d}:{:02d}:{:02d}".format(t1//3600, (t1//60) % 60, t1 % 60))
