# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Custom exceptions for Accordo validation."""


class AccordoError(Exception):
    """Base exception for all Accordo errors."""

    pass


class AccordoBuildError(AccordoError):
    """Raised when Accordo C++ library fails to build."""

    pass


class AccordoTimeoutError(AccordoError):
    """Raised when a kernel execution exceeds the timeout."""

    def __init__(self, message: str, timeout_seconds: float):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class AccordoProcessError(AccordoError):
    """Raised when the instrumented process crashes or fails."""

    def __init__(self, message: str, exit_code: int = None):
        super().__init__(message)
        self.exit_code = exit_code


class AccordoValidationError(AccordoError):
    """Raised when array validation fails."""

    pass
