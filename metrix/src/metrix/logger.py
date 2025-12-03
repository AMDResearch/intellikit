"""
Metrix logging utilities
"""

import logging


class MetrixLogger:
    """Logger that automatically prefixes all messages with [METRIX]"""

    def __init__(self, name: str = "metrix"):
        self._logger = logging.getLogger(name)

    def set_level(self, level: str):
        """
        Set logging level

        Args:
            level: Log level string (debug, info, warning, error)
        """
        log_level = getattr(logging, level.upper())
        logging.basicConfig(level=log_level, format="%(message)s")
        self._logger.setLevel(log_level)

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(f"[METRIX] {msg}", *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(f"[METRIX] {msg}", *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(f"[METRIX] {msg}", *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(f"[METRIX] {msg}", *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._logger.critical(f"[METRIX] {msg}", *args, **kwargs)


# Global logger instance
logger = MetrixLogger()

