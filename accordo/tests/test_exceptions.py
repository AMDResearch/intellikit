# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the Accordo exception hierarchy.

The hierarchy is load-bearing: ``cli._run_validate`` distinguishes an
``AccordoError`` (reported as a clean error payload) from any other exception
(reported as "Unexpected error"), so every custom exception must remain a
subclass of ``AccordoError``.
"""

import pytest

from accordo.exceptions import (
    AccordoBuildError,
    AccordoError,
    AccordoKernelNeverDispatched,
    AccordoProcessError,
    AccordoTimeoutError,
    AccordoValidationError,
)

SUBCLASSES = [
    AccordoBuildError,
    AccordoTimeoutError,
    AccordoProcessError,
    AccordoValidationError,
    AccordoKernelNeverDispatched,
]


class TestHierarchy:
    def test_base_is_an_exception(self):
        assert issubclass(AccordoError, Exception)

    @pytest.mark.parametrize("cls", SUBCLASSES, ids=lambda c: c.__name__)
    def test_every_error_derives_from_base(self, cls):
        assert issubclass(cls, AccordoError)

    @pytest.mark.parametrize("cls", SUBCLASSES, ids=lambda c: c.__name__)
    def test_catchable_as_base(self, cls):
        exc = cls("boom", 1) if cls in (AccordoTimeoutError, AccordoProcessError) else cls("boom")
        with pytest.raises(AccordoError):
            raise exc


class TestTimeoutError:
    def test_message_and_timeout_retained(self):
        exc = AccordoTimeoutError("took too long", 30.0)
        assert str(exc) == "took too long"
        assert exc.timeout_seconds == 30.0

    def test_timeout_requires_both_arguments(self):
        with pytest.raises(TypeError):
            AccordoTimeoutError("no timeout given")


class TestProcessError:
    def test_message_and_exit_code_retained(self):
        exc = AccordoProcessError("crashed", 139)
        assert str(exc) == "crashed"
        assert exc.exit_code == 139

    def test_exit_code_optional(self):
        exc = AccordoProcessError("crashed")
        assert exc.exit_code is None


class TestPlainSubclasses:
    @pytest.mark.parametrize(
        "cls",
        [AccordoBuildError, AccordoValidationError, AccordoKernelNeverDispatched],
        ids=lambda c: c.__name__,
    )
    def test_message_preserved(self, cls):
        assert str(cls("detail")) == "detail"
