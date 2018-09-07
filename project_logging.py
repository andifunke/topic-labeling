# -*- coding: utf-8 -*-

import logging
from constants import *


# create logger
logger = logging.getLogger('nlp_pipeline')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('../nlp_pipe.log')
fh.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)


def log(s: str, error: bool=False):
    if HPC or LOG:
        if not error:
            logger.info(s)
        else:
            logger.error(s)
    else:
        print(s)
