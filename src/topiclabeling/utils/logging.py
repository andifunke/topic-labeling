#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from datetime import datetime
from logging import (
    Formatter,
    CRITICAL,
    ERROR,
    WARNING,
    INFO,
    DEBUG,
    NOTSET,
    Logger,
    getLogger,
    FileHandler,
    StreamHandler,
)
from pathlib import Path
from pprint import pformat
from time import sleep
from typing import Union, Callable, Any

import pandas as pd
from tabulate import tabulate

import topiclabeling
from topiclabeling.utils.constants import LOG_DIR

LOGGER_NAME = ""
HAS_ERRORS = False
EXCEPTION = ERROR + 1


def timestamp() -> str:
    """Returns a formatted time stamp as string."""

    now = datetime.now()

    return now.strftime("%Y-%m-%d--%H-%M-%S--%f")


def init_logging(
    name: str = None,
    to_stdout: bool = False,
    to_file: bool = True,
    log_file: str = None,
    log_dir: str = None,
    append: bool = False,
    time_stamp: str = None,
    stdout_level=INFO,
    logfile_level=DEBUG,
    debug: bool = False,
) -> Logger:
    """
    Initialize a (global) logger for the current runtime.

    :param name: Give the logger a meaningful name. Log-file name will be referred from
        logger name.
    :param to_stdout: Set to True to print logging to standard-out.
    :param to_file: Set to True to write logging to a log-file.
    :param log_file: Name of a log-file (without path). By default referred from the
        logger name. Forces append=True.
    :param log_dir: Name of the log-dir. By default in ./PROJECT_ROOT/logs/.
    :param append: If True: use a single log-file for every run.
        If False: Create a unique, time-stamped log-file for each run.
    :param time_stamp: Pass a global time-stamp for the log file name. Only evaluated if
        append=False. If time_stamp is None, a new time-stamp will be created.
    :param stdout_level: Default: INFO
    :param logfile_level: Default: DEBUG
    :param debug: Set logging level to DEBUG for stdout and logfile.

    :returns: Logger
    """

    global LOGGER_NAME
    if LOGGER_NAME:
        logg("logging already initialized in this session", WARNING)
        return getLogger(LOGGER_NAME)

    name = name if name else "project"
    LOGGER_NAME = name

    if debug:
        logfile_level = stdout_level = DEBUG

    logger = getLogger(name)
    logger.setLevel(stdout_level)

    if to_file:
        if log_dir is None:
            log_dir = LOG_DIR
        else:
            log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if log_file:
            append = True
        if not append:
            if not time_stamp:
                time_stamp = timestamp()
            log_file = f"{name}__{time_stamp}.log"

        file_path = log_dir / log_file
        if not append:
            with open(file_path, "w"):
                target = log_file
                link_name = log_dir / f"latest__{name}.log"
                if link_name.exists():
                    link_name.unlink()
                link_name.symlink_to(target)

        fh = FileHandler(file_path)
        fh.setLevel(logfile_level)
        formatter_file = Formatter(
            "%(asctime)s--%(name)s--%(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter_file)
        logger.addHandler(fh)

    if to_stdout:
        ch = StreamHandler(sys.stdout)
        ch.setLevel(stdout_level)
        formatter_stdout = Formatter(
            "%(asctime)s--%(name)s--%(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter_stdout)
        logger.addHandler(ch)

    logger.info("")
    logger.info("#" * 75)
    logger.info(f"----- {name.upper()} -----")
    logger.info(f"- log-file: {log_file} -")
    logger.info("----- start -----")
    logger.debug(f"python: %s" % sys.version.replace("\n", " "))
    logger.debug(f"topiclabeling: {topiclabeling.__version__}")

    return logger


def logg(
    s: Any = "",
    level: int = INFO,
    logger: Logger = None,
    flush: bool = False,
) -> object:
    """
    Enhanced replacement for the print function. Falls back to print if necessary.

    Default logging level is INFO.

    :param s: String to log.
    :param level: INFO by default.
    :param logger: Instance of class logging.logger. Defaults to the last logger instance
        constructed by init_logging if a logger was already initialized.
    :param flush: Flushes StdOut and pauses the thread to avoid collision with other loggers or
        progress bar. Set it to False in case of frequent logging.
    """

    global HAS_ERRORS

    if flush:
        sleep(0.5)
        sys.stdout.flush()

    if not LOGGER_NAME and logger is None:
        print(s)
        return

    if not isinstance(s, str):
        s = str(s)

    try:
        if logger is None:
            logger = getLogger(LOGGER_NAME)

        level = level if isinstance(level, int) else None
        if level is None or level == INFO:
            logger.info(s)
        elif level == NOTSET:
            pass
        elif level == DEBUG:
            logger.debug(s)
        elif level == WARNING:
            logger.warning(s)
        elif level == ERROR:
            logger.error(s)
            HAS_ERRORS = True
        elif level == EXCEPTION:
            logger.exception(s)
            HAS_ERRORS = True
        elif level == CRITICAL:
            logger.critical(s, exc_info=True)
            HAS_ERRORS = True
        else:
            print(s)
    except Exception as e:
        print(e)
        print(s)
    finally:
        if flush:
            sleep(0.5)
            sys.stdout.flush()


def logging_frame(
    main_function: Callable,
    kwargs: dict,
    project_name: str = None,
    develop_mode: bool = False,
    time_stamp: str = None,
):
    """
    Sets up a logging frame to deploy modules and catch critical events.

    It initializes a logger and takes care of logging fatal exceptions.

    :param main_function: Pass the main function of your module.
    :param kwargs: A dict with key-value pairs of the parameters to the main function.
    :param project_name: An optional name of the project, mainly used for logging.
    :param develop_mode: bool. Activate develop mode and debug logging.
    :param time_stamp: str. Provide ane external time_stamp for naming the logging file.
    """

    init_logging(
        name=project_name,
        to_stdout=True,
        to_file=True,
        debug=develop_mode,
        time_stamp=time_stamp,
    )
    try:
        main_function(**kwargs)
    except Exception as e:
        logg(e, CRITICAL)
    else:
        if HAS_ERRORS:
            logg("Some Errors occurred.", WARNING)
        else:
            logg("No Errors occurred.", DEBUG)
    finally:
        logg("----- exit -----", WARNING)


def table(
    df: Union[pd.DataFrame, pd.Series], head: int = 0, floatfmt: str = None
) -> str:
    """Generates a well formatted table from a DataFrame or Series."""

    if not (isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)):
        logg("generate table: Not a DataFrame.", WARNING)
        return ""

    if head > 0:
        df = df.head(head)
    elif head < 0:
        df = df.tail(-head)

    kwargs = dict()
    if floatfmt is not None:
        kwargs["floatfmt"] = floatfmt

    try:
        tbl = tabulate(
            df, headers="keys", tablefmt="pipe", showindex="always", **kwargs
        )
        return tbl
    except Exception as e:
        logg(e)
        logg(df)


def tprint(
    df: pd.DataFrame,
    head: int = 0,
    floatfmt: str = None,
    to_latex: bool = False,
    log_shape: bool = True,
    message: str = "",
):
    """Print a DataFrame as a well formatted table."""

    if df is None:
        return
    tbl = table(df=df, head=head, floatfmt=floatfmt)
    shape = f"\nshape: {df.shape}" if log_shape else ""
    logg(f"{message}\n{tbl}{shape}")

    if to_latex:
        print(df.to_latex(bold_rows=True))


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60

    return f"{h}:{m:>02}:{s:>05.2f}"


def log_args(args):
    logg("\n" + pformat(vars(args)))
