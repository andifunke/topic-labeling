# -*- coding: utf-8 -*-

import re
from os import listdir
from os.path import isfile, join
from time import time

from topiclabeling.utils.utils import init_logging
from topiclabeling.utils.options import CORPUS_PREFIXES, DE, STORE, START, BATCH_SIZE, BATCHES
from topiclabeling.utils.constants import ETL_DIR
from topiclabeling.preprocessing.nlp_processor import NLProcessor


if __name__ == "__main__":
    t0 = time()

    logger = init_logging('NLP', to_stdout=True, to_file=False)

    def logg(msg):
        logger.info(msg)

    logg("##### START #####")

    # filter files for certain prefixes
    prefixes = r'^(' + '|'.join(CORPUS_PREFIXES) + r').'
    pattern = re.compile(prefixes)
    files = sorted([
        f for f in listdir(ETL_DIR)
        if (isfile(join(ETL_DIR, f)) and pattern.match(f))
    ])
    processor = NLProcessor(spacy_path=DE, log_fn=logg)

    start = START  # 550_000
    batch_size = BATCH_SIZE  # 50_000
    batches = BATCHES

    for name in files:
        corpus = name.split('.')[0]
        filename = join(ETL_DIR, name)
        for i in range(1, batches+1):
            logg(f">>> batch: {i:d} >>>")
            processor.read_process_store(
                filename, corpus,
                start=start,
                stop=(start+batch_size) if batch_size else None,
            )
            if batch_size:
                start += batch_size
            else:
                break

    t1 = int(time() - t0)
    logg(f"all done in {t1//3600:02d}:{(t1//60) % 60:02d}:{t1 % 60:02d}")
