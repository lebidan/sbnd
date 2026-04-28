# Misc utility functions for the SBND project

import logging
from lightning.pytorch.utilities import rank_zero_only


def get_rank_zero_logger(name: str = __name__) -> logging.Logger:
    """
    Initializes multi-GPU-friendly python command line logger.
    From: https://github.com/gorodnitskiy/yet-another-lightning-hydra-template/
    """

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


if __name__ == "__main__":
    pass
