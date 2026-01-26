# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

from rocm_mcp import HipDocs


def test_hip_docs_initialization() -> None:
    """Test HIP docs initialization with default version."""
    hip_docs = HipDocs()
    assert hip_docs.version == "latest"
    assert "rocm.docs.amd.com" in hip_docs.base_url


def test_hip_docs_initialization_with_version() -> None:
    """Test HIP docs initialization with specific version."""
    hip_docs = HipDocs(version="6.0")
    assert hip_docs.version == "6.0"
    assert "6.0" in hip_docs.base_url


def test_hip_docs_search_api() -> None:
    """Test searching HIP API documentation."""
    hip_docs = HipDocs()
    results = hip_docs.search_api("hip", limit=3)
    # We expect some results, but exact count may vary
    assert isinstance(results, list)
    # Each result should have required fields
    for result in results:
        assert hasattr(result, "title")
        assert hasattr(result, "url")
        assert hasattr(result, "description")


def test_hip_docs_search_api_no_results() -> None:
    """Test searching with query that likely has no results."""
    hip_docs = HipDocs()
    results = hip_docs.search_api("xyznonexistentfunction123", limit=5)
    # Should return empty list, not raise an error
    assert isinstance(results, list)
