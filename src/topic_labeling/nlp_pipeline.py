# -*- coding: utf-8 -*-

import re
from os import listdir
from os.path import isfile, join
from time import time

from topic_labeling.utils import init_logging

from topic_labeling import options
options.update_from_args()
from topic_labeling.constants import FULL_PATH
from topic_labeling.nlp_processor import NLPProcessor
from topic_labeling.options import CORPUS_PREFIXES, DE, STORE, START, BATCH_SIZE, BATCHES


if __name__ == "__main__":
    t0 = time()

    ### --- run ---
    logger = init_logging('NLP')

    def logg(msg):
        logger.info(msg)

    logg("##### START #####")

    # filter files for certain prefixes
    prefixes = r'^(' + '|'.join(CORPUS_PREFIXES) + r').'
    pattern = re.compile(prefixes)
    files = sorted([
        f for f in listdir(FULL_PATH)
        if (isfile(join(FULL_PATH, f)) and pattern.match(f))
    ])
    processor = NLPProcessor(spacy_path=DE, logg=logg)

    start = START  # 550_000
    batch_size = BATCH_SIZE  # 50_000
    batches = BATCHES

    for name in files:
        corpus = name.split('.')[0]
        fname = join(FULL_PATH, name)
        for i in range(1, batches+1):
            logg('>>> batch: {:d} >>>'.format(i))
            processor.read_process_store(
                fname, corpus,
                start=start,
                stop=(start+batch_size) if batch_size else None,
                store=STORE,
                # vocab_to_disk=STORE,
                # print=True,
                # head=1000,
            )
            if batch_size:
                start += batch_size
            else:
                break

    t1 = int(time() - t0)
    logg("all done in {:02d}:{:02d}:{:02d}".format(t1//3600, (t1//60) % 60, t1 % 60))
