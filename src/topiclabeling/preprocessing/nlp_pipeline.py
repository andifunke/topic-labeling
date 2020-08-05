# -*- coding: utf-8 -*-

import re
from time import time

from topiclabeling.preprocessing.nlp_processor import NLProcessor
from topiclabeling.utils.args import global_args
from topiclabeling.utils.constants import ETL_DIR
from topiclabeling.utils.logging import init_logging, logg

if __name__ == "__main__":
    t0 = time()

    args = global_args()
    init_logging('NLP', to_stdout='stdout' in args.log, to_file='file' in args.log)

    logg("##### START #####")

    # filter files for certain prefixes
    prefixes = r'^(' + '|'.join(args.corpus) + r').'
    pattern = re.compile(prefixes)
    files = sorted(f for f in ETL_DIR.iterdir() if f.is_file() and pattern.match(f.name))
    processor = NLProcessor(spacy_path=args.spacy_path)

    start = args.start

    for file in files:
        corpus = file.name.split('.')[0]
        for i in range(1, args.batches + 1):
            logg(f">>> batch: {i:d} >>>")
            processor.read_process_store(
                file, corpus,
                start=start,
                stop=(start + args.batch_size) if args.batch_size else None,
            )
            if args.batch_size:
                start += args.batch_size

    t1 = int(time() - t0)
    logg(f"all done in {t1//3600:02d}:{(t1//60) % 60:02d}:{t1 % 60:02d}")
