# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# This setup.py provides backward compatibility for legacy metadata fields
# that don't map directly from pyproject.toml's modern PEP 621 format.
setup(
    name="metrix",
    description="GPU Profiling. Decoded. Clean metrics for humans, not hardware counters for engineers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AMDResearch/intellikit/tree/main/metrix",
    author="Muhammad Awad",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=1.5.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Profiling",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        "console_scripts": [
            "metrix=metrix.cli.main:main",
        ],
    },
)
