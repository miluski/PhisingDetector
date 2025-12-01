import logging
import sys

from config.colored_formatter import ColoredFormatter
from config.operation_logger import OperationLogger


def setup_logging(level: str = "INFO") -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if root_logger.handlers:
        root_logger.handlers.clear()

    handler = _create_console_handler()
    root_logger.addHandler(handler)
    return root_logger


def _create_console_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    return handler


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


setup_logging("DEBUG")
