# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

version_number = "0.1.0"

html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "hyperloom",
}

# for PDF output on Read the Docs
project = "IntelliKit"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."  # noqa: A001

external_toc_path = "./sphinx/_toc.yml"

# Add more additional package accordingly
extensions = [
    "rocm_docs",
]

html_title = f"{project} {version_number} documentation"

external_projects_current_project = "IntelliKit"
