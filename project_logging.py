# -*- coding: utf-8 -*-

import logging
from sys import stdout
from os import makedirs
from os.path import dirname
from options import HPC, LOG, NOTEBOOK, LOG_PATH

# create path if necessary
makedirs(dirname(LOG_PATH), exist_ok=True)

# create logger
logger = logging.getLogger('Topic_labeling')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)

# logging to file
if LOG or HPC:
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# logging to stdout
if not (HPC or NOTEBOOK):
    ch = logging.StreamHandler(stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def log(s: str, error: bool=False):
    if error:
        logger.error(s)
    else:
        logger.info(s)
