# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

import logging
from dataclasses import dataclass
from typing import ClassVar

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
    _cache: ClassVar[dict[str, list[HipApiResult]]] = {}

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

    def _fetch_index(self) -> list[HipApiResult]:
        """Fetch and parse the API index."""
        if self.version in self._cache:
            return self._cache[self.version]

        self.logger.info("Fetching HIP API index for version %s", self.version)
        results = []
        try:
            url = f"{self.base_url}/genindex.html"
            with httpx.Client(follow_redirects=True, timeout=30.0) as client:
                response = client.get(url)
                response.raise_for_status()

                # Parse the HTML to find relevant API functions
                soup = BeautifulSoup(response.text, "html.parser")

                # Find all links in the index tables
                tables = soup.find_all("table", class_="genindextable")

                for table in tables:
                    links = table.find_all("a", href=True)
                    for link in links:
                        link_text = link.get_text().strip()
                        href = link["href"]

                        if isinstance(href, list):
                            href = href[0]

                        # Construct full URL
                        if not href.startswith("http"):
                            full_url = f"{self.base_url}/{href}"
                        else:
                            full_url = href

                        # Get description (use title as description since genindex lacks context)
                        description = link_text

                        results.append(
                            HipApiResult(
                                title=link_text,
                                url=full_url,
                                description=description,
                                content=None,
                            )
                        )

            self._cache[self.version] = results
            self.logger.info("Cached %d API entries for version %s", len(results), self.version)

        except httpx.HTTPError as e:
            self.logger.exception("Failed to fetch HIP API index: %s", str(e))
        except Exception as e:
            self.logger.exception("Unexpected error while fetching HIP API index: %s", str(e))

        return results

    def search_api(self, query: str, limit: int = 5) -> list[HipApiResult]:
        """Search HIP API documentation.

        Args:
            query (str): Search query for API documentation.
            limit (int): Maximum number of results to return. Defaults to 5.

        Returns:
            list[HipApiResult]: List of API documentation results.
        """
        self.logger.info("Searching HIP API documentation for query: %s", query)

        all_results = self._fetch_index()
        query_lower = query.lower()

        matches = []
        for result in all_results:
            if query_lower in result.title.lower():
                matches.append(result)
                if len(matches) >= limit:
                    break

        self.logger.info("Found %d results for query: %s", len(matches), query)
        return matches

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
