# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

import logging
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from rocm_mcp.doc.hip_docs import HipDocs

# initialize server
mcp = FastMCP(
    name="hip_docs",
    instructions=(
        "MCP server for accessing HIP (Heterogeneous-computing Interface for Portability) "
        "language and runtime developer reference documentation."
    ),
)
logger = logging.getLogger(mcp.name)


@mcp.tool()
def search_hip_api(
    query: Annotated[str, Field(description="Search query for HIP API documentation.")],
    version: Annotated[
        str, Field(description="HIP version to search. Defaults to 'latest'.")
    ] = "latest",
    limit: Annotated[
        int, Field(description="Maximum number of results to return. Defaults to 5.")
    ] = 5,
) -> str:
    """Search HIP API documentation for functions, classes, and other API references.

    This tool searches the official HIP documentation at rocm.docs.amd.com for API
    documentation matching the query. It can search for functions like hipMalloc,
    hipMemcpy, classes, and other HIP API elements.

    Returns:
        str: Formatted search results with titles, URLs, and descriptions.
    """
    try:
        hip_docs = HipDocs(version=version)
        results = hip_docs.search_api(query, limit=limit)

        if not results:
            return f"No HIP API documentation found for query: {query}"

        lines = [f"Found {len(results)} results for '{query}' in HIP {version} documentation:"]
        lines.append("")

        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.title}")
            lines.append(f"   URL: {result.url}")
            lines.append(f"   Description: {result.description}")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        logger.exception("Failed to search HIP API documentation: %s", str(e))
        return f"Error searching HIP API documentation: {e!s}"


@mcp.tool()
def get_hip_api_reference(
    api_name: Annotated[str, Field(description="Name of the HIP API function or class.")],
    version: Annotated[
        str, Field(description="HIP version to use. Defaults to 'latest'.")
    ] = "latest",
) -> str:
    """Get detailed reference documentation for a specific HIP API function or class.

    This tool retrieves comprehensive documentation for a specific HIP API element,
    including its full description, parameters, return values, and usage examples.

    Returns:
        str: Detailed API reference documentation.
    """
    try:
        hip_docs = HipDocs(version=version)
        result = hip_docs.get_api_reference(api_name)

        if not result:
            return f"No HIP API reference found for: {api_name}"

        lines = [f"HIP API Reference: {result.title}"]
        lines.append("")
        lines.append(f"URL: {result.url}")
        lines.append("")
        lines.append("Description:")
        lines.append(result.description)

        if result.content:
            lines.append("")
            lines.append("Full Documentation:")
            lines.append(result.content)

        return "\n".join(lines)
    except Exception as e:
        logger.exception("Failed to get HIP API reference: %s", str(e))
        return f"Error getting HIP API reference: {e!s}"


def main() -> None:
    """Main function to run the HIP documentation MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
