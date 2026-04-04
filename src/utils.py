import logging, sys
import colorlog  # type: ignore[import-untyped]
from pytorch_lightning.utilities import rank_zero_only


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


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with the same colorlog format used by Hydra's job_logging."""
    handler = colorlog.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        colorlog.ColoredFormatter(
            fmt="[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            log_colors={
                "DEBUG": "purple",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            },
        )
    )
    logging.root.setLevel(level)
    logging.root.addHandler(handler)


if __name__ == "__main__":
    pass
