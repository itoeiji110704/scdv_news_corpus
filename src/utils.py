import copy
import logging
import os
import random
import time

import numpy as np

from src.constants import LOGGER_NAME


class ColoredStreamHandler(logging.StreamHandler):
    """Colored stream handler for logging."""

    cmap = {
        "TRACE": "[TRACE]",
        "DEBUG": "\x1b[0;36mDEBUG\x1b[0m",
        "INFO": "\x1b[0;32mINFO\x1b[0m",
        "WARNING": "\x1b[0;33mWARN\x1b[0m",
        "WARN": "\x1b[0;33mwWARN\x1b[0m",
        "ERROR": "\x1b[0;31mERROR\x1b[0m",
        "ALERT": "\x1b[0;37;41mALERT\x1b[0m",
        "CRITICAL": "\x1b[0;37;41mCRITICAL\x1b[0m",
    }

    def emit(self, record: logging.LogRecord) -> None:
        record = copy.deepcopy(record)
        record.levelname = self.cmap[record.levelname]
        super().emit(record)


def _setup_logger():
    """Setup logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s %(lineno)d %(message)s",
        datefmt="%Y/%m/%d %I:%M:%S",
    )
    stream_handler = ColoredStreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


logger = _setup_logger()


def timeit(f):
    """A simple decorator for measuring time of a function call."""

    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info(f"Measuring time for {f}")
        res = f(*args, **kwargs)
        end = time.time()
        logger.info(f"{f} took {end - start:.5f} seconds")
        return res

    return wrapper


def fix_seed(seed):
    """Fix random seeds."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
