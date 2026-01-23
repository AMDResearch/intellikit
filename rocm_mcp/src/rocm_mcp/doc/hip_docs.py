# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

import logging
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

# Constants
MAX_DESCRIPTION_LENGTH = 500


@dataclass(frozen=True)
class HipApiResult:
    """Result of HIP API documentation search.

    Attributes:
        title (str): Title of the API documentation.
        url (str): URL to the full documentation.
        description (str): Brief description of the API.
        content (str | None): Full content of the documentation if available.
    """

    title: str
    url: str
    description: str
    content: str | None = None


class HipDocs:
    """Class to fetch and search HIP documentation.

    This class provides methods to access HIP (Heterogeneous-computing Interface for Portability)
    API documentation from AMD ROCm documentation.

    Example:
        Basic usage::

            from omnikit import HipDocs

            # Initialize HIP docs accessor
            hip_docs = HipDocs()

            # Search for API documentation
            results = hip_docs.search_api("hipMalloc")
            for result in results:
                print(f"{result.title}: {result.description}")

        With specific HIP version::

            # Use a specific HIP version
            hip_docs = HipDocs(version="6.0")
            results = hip_docs.search_api("hipMemcpy")

    Attributes:
        logger (logging.Logger): Logger for logging documentation access.
        version (str): HIP version to use for documentation.
        base_url (str): Base URL for HIP documentation.
    """

    logger: logging.Logger
    version: str
    base_url: str

    def __init__(self, logger: logging.Logger | None = None, version: str = "latest") -> None:
        """Initialize the HipDocs accessor.

        Args:
            logger (logging.Logger | None): Logger for logging. If None, a default logger is
                created.
            version (str): HIP version to use. Defaults to "latest".
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.version = version
        self.base_url = f"https://rocm.docs.amd.com/projects/HIP/en/{version}"
        self.logger.info("Initialized HIP documentation accessor for version %s", version)

    def search_api(self, query: str, limit: int = 5) -> list[HipApiResult]:
        """Search HIP API documentation.

        Args:
            query (str): Search query for API documentation.
            limit (int): Maximum number of results to return. Defaults to 5.

        Returns:
            list[HipApiResult]: List of API documentation results.
        """
        self.logger.info("Searching HIP API documentation for query: %s", query)
        results = []

        try:
            # First, try to get the API reference index page
            url = f"{self.base_url}/doxygen/html/index.html"
            with httpx.Client(follow_redirects=True, timeout=30.0) as client:
                response = client.get(url)
                response.raise_for_status()

                # Parse the HTML to find relevant API functions
                soup = BeautifulSoup(response.text, "html.parser")

                # Find all links that might be API references
                links = soup.find_all("a", href=True)
                query_lower = query.lower()

                for link in links:
                    link_text = link.get_text().strip()
                    href = link["href"]

                    if query_lower in link_text.lower() and len(results) < limit:
                        # Construct full URL
                        if not href.startswith("http"):
                            full_url = f"{self.base_url}/doxygen/html/{href}"
                        else:
                            full_url = href

                        # Get description from surrounding context
                        description = link_text
                        parent = link.parent
                        if parent:
                            parent_text = parent.get_text().strip()
                            if len(parent_text) > len(link_text):
                                description = parent_text[:MAX_DESCRIPTION_LENGTH]

                        results.append(
                            HipApiResult(
                                title=link_text,
                                url=full_url,
                                description=description,
                                content=None,
                            )
                        )

        except httpx.HTTPError as e:
            self.logger.exception("Failed to search HIP API documentation: %s", str(e))
        except Exception as e:
            self.logger.exception("Unexpected error while searching HIP API: %s", str(e))

        self.logger.info("Found %d results for query: %s", len(results), query)
        return results

    def get_api_reference(self, api_name: str) -> HipApiResult | None:
        """Get detailed API reference for a specific HIP API function or class.

        Args:
            api_name (str): Name of the API function or class.

        Returns:
            HipApiResult | None: API reference details or None if not found.
        """
        self.logger.info("Getting API reference for: %s", api_name)

        # Use search_api as the primary method since direct URL patterns are unreliable
        results = self.search_api(api_name, limit=1)
        if results:
            return results[0]

        self.logger.warning("No API reference found for: %s", api_name)
        return None
