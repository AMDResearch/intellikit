# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Samplex logging utilities
"""

import logging


class SamplexLogger:
    """Logger that automatically prefixes all messages with [SAMPLEX]"""

    def __init__(self, name: str = "samplex"):
        self._logger = logging.getLogger(name)

    def set_level(self, level: str):
        log_level = getattr(logging, level.upper())
        logging.basicConfig(level=log_level, format="%(message)s")
        self._logger.setLevel(log_level)

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(f"[SAMPLEX] {msg}", *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(f"[SAMPLEX] {msg}", *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(f"[SAMPLEX] {msg}", *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(f"[SAMPLEX] {msg}", *args, **kwargs)


# Global logger instance
logger = SamplexLogger()
