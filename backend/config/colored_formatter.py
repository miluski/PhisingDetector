"""Colored logging formatter for interactive backend output."""

import logging


class ColoredFormatter(logging.Formatter):
    """Add ANSI colors to level and logger names."""
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        """Format a record while preserving the standard formatter contract."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{self.BOLD}{record.levelname}{self.RESET}"
        record.name = f"\033[34m{record.name}{self.RESET}"
        return super().format(record)
