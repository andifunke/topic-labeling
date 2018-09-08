# -*- coding: utf-8 -*-

import logging
from sys import stdout
from constants import HPC, LOG, LOG_PATH

# create logger
logger = logging.getLogger('NLP_pipe')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')

if LOG or HPC:
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

if not HPC:
    # stdout logger
    ch = logging.StreamHandler(stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def log(s: str, error: bool=False):
    if error:
        logger.error(s)
    else:
        logger.info(s)
