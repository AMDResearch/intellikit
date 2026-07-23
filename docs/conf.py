# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

version_number = "0.1.0"

html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "hyperloom",
    "header_title": f"IntelliKit {version_number}",
    "header_link": False,
    "nav_secondary_items": {
        "Hyperloom": "https://rocm.docs.amd.com/projects/hyperloom/en/latest/index.html",
        "GitHub": "https://github.com/AMDResearch/intellikit",
        "Blogs": "https://rocm.blogs.amd.com/",
        "ROCm Developer Hub": "https://www.amd.com/en/developer/resources/rocm-hub.html",
        "Instinct Docs": "https://instinct.docs.amd.com/",
        "Infinity Hub": "https://www.amd.com/en/developer/resources/infinity-hub.html",
        "Support": "https://github.com/AMDResearch/intellikit/issues/new/choose",
    },
    "link_main_doc": False,
}

# for PDF output on Read the Docs
project = "IntelliKit"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."  # noqa: A001
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"

# Add more additional package accordingly
extensions = [
    "rocm_docs",
]

html_title = f"{project} {version_number} documentation"

external_projects_current_project = "IntelliKit"
