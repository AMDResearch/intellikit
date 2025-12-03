"""
Metrix setup configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="metrix",
    version="0.1.0",
    author="AMD",
    description="GPU Profiling. Decoded. Clean metrics for humans, not hardware counters for engineers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amd/metrix",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Profiling",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
    ],
    entry_points={
        "console_scripts": [
            "metrix=metrix.cli.main:main",
        ],
    },
)

