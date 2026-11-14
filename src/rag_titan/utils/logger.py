"""
Logging helpers for the whole package.

A library should never force log output on the people who use it,
so `get_logger()` only creates a logger. Call `setup_logging()` once
in your own script (or use the CLI) when you want to see the logs.

Example:
    from rag_titan import setup_logging
    setup_logging("DEBUG")
"""

import logging
import os

# name of the top level logger, every module logger is a child of it
PACKAGE_LOGGER_NAME = "rag_titan"

# stop "No handler found" warnings when the user has not set up logging
logging.getLogger(PACKAGE_LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name):
    """
    Return a logger for one module.

    Args:
        name: usually `__name__` of the calling module.

    Returns:
        A standard `logging.Logger`.
    """
    return logging.getLogger(name)


def setup_logging(level=None):
    """
    Print rag_titan logs to the terminal.

    Args:
        level: "DEBUG", "INFO" or "WARNING". When empty, the LOG_LEVEL
            environment variable is used, and INFO is the default.
    """
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger(PACKAGE_LOGGER_NAME).setLevel(level)

    # hide every HTTP request line printed by the ollama client
    logging.getLogger("httpx").setLevel(logging.WARNING)
