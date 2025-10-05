"""
J.A.R.V.I.S. Web Searcher Module
Advanced web scraping and API-based search for AI research
"""

import os
import time
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import logging

# Import search libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


class WebSearcher:
    """
    Advanced web search and scraping system
    Uses multiple search APIs and scraping techniques for comprehensive research
    """

    def __init__(self, jarvis_instance):
        """
        Initialize web searcher

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.WebSearcher')

        # Search configuration
        self.search_apis = {
            "google": {
                "enabled": True,
                "api_key": os.getenv("GOOGLE_SEARCH_API_KEY"),
                "cse_id": os.getenv("GOOGLE_CSE_ID"),
                "base_url": "https://www.googleapis.com/customsearch/v1"
            },
            "bing": {
                "enabled": True,
                "api_key": os.getenv("BING_SEARCH_API_KEY"),
                "base_url": "https://api.bing.microsoft.com/v7.0/search"
            },
            "duckduckgo": {
                "enabled": True,
                "base_url": "https://api.duckduckgo.com/"
            },
            "serpapi": {
                "enabled": True,
                "api_key": os.getenv("SERPAPI_KEY"),
                "base_url": "https://serpapi.com/search.json"
            }
        }

        # Scraping settings
        self.scraping_config = {
            "max_pages_per_site": 5,
            "request_timeout": 10,
            "max_content_length": 50000,
            "respect_robots_txt": True,
            "user_agent": "J.A.R.V.I.S. Research Bot/2.0"
        }

        # Rate limiting
        self.rate_limits = {
            "requests_per_minute": 60,
            "last_request_time": 0,
            "request_count": 0
        }

        # Cache for search results
        self.search_cache = {}
        self.cache_ttl = 3600  # 1 hour

    async def initialize(self):
        """Initialize web searcher"""
        try:
            self.logger.info("Initializing web searcher...")

            # Test API connections
            await self._test_api_connections()

            self.logger.info("Web searcher initialized")

        except Exception as e:
            self.logger.error(f"Error initializing web searcher: {e}")
            raise

    async def _test_api_connections(self):
        """Test connections to search APIs"""
        for api_name, config in self.search_apis.items():
            if config["enabled"] and config.get("api_key"):
                try:
                    # Test API key validity
                    test_result = await self._test_api_key(api_name, config)
                    if test_result:
                        self.logger.info(f"✓ {api_name} API connection successful")
                    else:
                        self.logger.warning(f"✗ {api_name} API connection failed")
                        config["enabled"] = False

                except Exception as e:
                    self.logger.error(f"Error testing {api_name} API: {e}")
                    config["enabled"] = False

    async def _test_api_key(self, api_name: str, config: Dict[str, Any]) -> bool:
        """Test API key validity"""
        try:
            if api_name == "google":
                return await self._test_google_api(config)
            elif api_name == "bing":
                return await self._test_bing_api(config)
            elif api_name == "serpapi":
                return await self._test_serpapi(config)

            return True

        except Exception as e:
            self.logger.error(f"Error testing {api_name} API key: {e}")
            return False

    async def _test_google_api(self, config: Dict[str, Any]) -> bool:
        """Test Google Custom Search API"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "key": config["api_key"],
                    "cx": config["cse_id"],
                    "q": "test",
                    "num": 1
                }

                async with session.get(config["base_url"], params=params) as response:
                    return response.status == 200

        except Exception as e:
            self.logger.error(f"Google API test failed: {e}")
            return False

    async def _test_bing_api(self, config: Dict[str, Any]) -> bool:
        """Test Bing Search API"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Ocp-Apim-Subscription-Key": config["api_key"]}
                params = {"q": "test", "count": 1}

                async with session.get(config["base_url"], headers=headers, params=params) as response:
                    return response.status == 200

        except Exception as e:
            self.logger.error(f"Bing API test failed: {e}")
            return False

    async def _test_serpapi(self, config: Dict[str, Any]) -> bool:
        """Test SerpAPI"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "engine": "google",
                    "api_key": config["api_key"],
                    "q": "test",
                    "num": 1
                }

                async with session.get(config["base_url"], params=params) as response:
                    return response.status == 200

        except Exception as e:
            self.logger.error(f"SerpAPI test failed: {e}")
            return False

    async def search(self,
                    query: str,
                    max_results: int = 20,
                    include_content: bool = True,
                    search_apis: List[str] = None) -> List[Dict[str, Any]]:
        """
        Perform comprehensive web search

        Args:
            query: Search query
            max_results: Maximum number of results
            include_content: Whether to scrape page content
            search_apis: Specific APIs to use

        Returns:
            List of search results
        """
        try:
            # Check cache first
            cache_key = f"{query}:{max_results}:{include_content}"
            if cache_key in self.search_cache:
                cached_time, cached_results = self.search_cache[cache_key]
                if time.time() - cached_time < self.cache_ttl:
                    self.logger.info(f"Using cached results for query: {query}")
                    return cached_results

            # Determine which APIs to use
            if search_apis is None:
                search_apis = [name for name, config in self.search_apis.items() if config["enabled"]]

            all_results = []

            # Execute searches concurrently
            search_tasks = []

            for api_name in search_apis:
                if api_name in self.search_apis and self.search_apis[api_name]["enabled"]:
                    task = asyncio.create_task(
                        self._search_single_api(api_name, query, max_results // len(search_apis))
                    )
                    search_tasks.append((api_name, task))

            # Wait for all searches to complete
            for api_name, task in search_tasks:
                try:
                    results = await asyncio.wait_for(task, timeout=30)
                    all_results.extend(results)

                    self.logger.info(f"API {api_name} returned {len(results)} results")

                except asyncio.TimeoutError:
                    self.logger.warning(f"API {api_name} search timed out")
                except Exception as e:
                    self.logger.error(f"Error searching {api_name}: {e}")

            # Remove duplicates and rank results
            unique_results = self._deduplicate_results(all_results)
            ranked_results = self._rank_results(unique_results, query)

            # Scrape content if requested
            if include_content:
                await self._scrape_content(ranked_results)

            # Cache results
            self.search_cache[cache_key] = (time.time(), ranked_results)

            self.logger.info(f"Web search completed: {len(ranked_results)} total results for '{query}'")

            return ranked_results[:max_results]

        except Exception as e:
            self.logger.error(f"Error performing web search: {e}")
            return []

    async def _search_single_api(self, api_name: str, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using a single API"""
        try:
            config = self.search_apis[api_name]

            if api_name == "google":
                return await self._search_google(config, query, max_results)
            elif api_name == "bing":
                return await self._search_bing(config, query, max_results)
            elif api_name == "duckduckgo":
                return await self._search_duckduckgo(config, query, max_results)
            elif api_name == "serpapi":
                return await self._search_serpapi(config, query, max_results)
            else:
                return []

        except Exception as e:
            self.logger.error(f"Error searching {api_name}: {e}")
            return []

    async def _search_google(self, config: Dict[str, Any], query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "key": config["api_key"],
                    "cx": config["cse_id"],
                    "q": query,
                    "num": min(max_results, 10),  # Google API limit
                    "safe": "active"
                }

                async with session.get(config["base_url"], params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        results = []
                        for item in data.get("items", []):
                            result = {
                                "title": item.get("title", ""),
                                "url": item.get("link", ""),
                                "snippet": item.get("snippet", ""),
                                "source": "google",
                                "timestamp": time.time()
                            }
                            results.append(result)

                        return results
                    else:
                        self.logger.error(f"Google API error: {response.status}")
                        return []

        except Exception as e:
            self.logger.error(f"Error searching Google: {e}")
            return []

    async def _search_bing(self, config: Dict[str, Any], query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Bing Search API"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Ocp-Apim-Subscription-Key": config["api_key"]}
                params = {
                    "q": query,
                    "count": min(max_results, 50),
                    "safeSearch": "Strict"
                }

                async with session.get(config["base_url"], headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        results = []
                        for item in data.get("webPages", {}).get("value", []):
                            result = {
                                "title": item.get("name", ""),
                                "url": item.get("url", ""),
                                "snippet": item.get("snippet", ""),
                                "source": "bing",
                                "timestamp": time.time()
                            }
                            results.append(result)

                        return results
                    else:
                        self.logger.error(f"Bing API error: {response.status}")
                        return []

        except Exception as e:
            self.logger.error(f"Error searching Bing: {e}")
            return []

    async def _search_duckduckgo(self, config: Dict[str, Any], query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo (scraping)"""
        try:
            # DuckDuckGo doesn't have a public API, so we scrape
            url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"

            async with aiohttp.ClientSession() as session:
                headers = {"User-Agent": self.scraping_config["user_agent"]}

                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()

                        soup = BeautifulSoup(html, 'html.parser')
                        results = []

                        for result in soup.find_all('div', class_='result')[:max_results]:
                            title_elem = result.find('a', class_='result__a')
                            snippet_elem = result.find('a', class_='result__snippet')

                            if title_elem:
                                result_item = {
                                    "title": title_elem.get_text(),
                                    "url": title_elem.get('href', ""),
                                    "snippet": snippet_elem.get_text() if snippet_elem else "",
                                    "source": "duckduckgo",
                                    "timestamp": time.time()
                                }
                                results.append(result_item)

                        return results

        except Exception as e:
            self.logger.error(f"Error searching DuckDuckGo: {e}")
            return []

    async def _search_serpapi(self, config: Dict[str, Any], query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using SerpAPI"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "engine": "google",
                    "api_key": config["api_key"],
                    "q": query,
                    "num": min(max_results, 100)
                }

                async with session.get(config["base_url"], params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        results = []
                        for item in data.get("organic_results", []):
                            result = {
                                "title": item.get("title", ""),
                                "url": item.get("link", ""),
                                "snippet": item.get("snippet", ""),
                                "source": "serpapi",
                                "timestamp": time.time()
                            }
                            results.append(result)

                        return results
                    else:
                        self.logger.error(f"SerpAPI error: {response.status}")
                        return []

        except Exception as e:
            self.logger.error(f"Error searching SerpAPI: {e}")
            return []

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results"""
        seen_urls = set()
        unique_results = []

        for result in results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)

        return unique_results

    def _rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank search results by relevance"""
        query_terms = set(query.lower().split())

        for result in results:
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()

            # Calculate relevance score
            score = 0

            # Title matches get higher score
            for term in query_terms:
                if term in title:
                    score += 3
                elif term in snippet:
                    score += 1

            # Exact phrase matches get bonus
            if query.lower() in title:
                score += 5

            result["relevance_score"] = score

        # Sort by relevance score
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        return results

    async def _scrape_content(self, results: List[Dict[str, Any]]):
        """Scrape content from result URLs"""
        try:
            async with aiohttp.ClientSession() as session:
                for result in results:
                    url = result.get("url", "")
                    if url and len(result.get("snippet", "")) < 200:  # Only scrape if snippet is short
                        try:
                            content = await self._scrape_single_page(session, url)
                            if content:
                                result["content"] = content
                                result["content_scraped"] = True

                        except Exception as e:
                            self.logger.debug(f"Error scraping {url}: {e}")

        except Exception as e:
            self.logger.error(f"Error scraping content: {e}")

    async def _scrape_single_page(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Scrape content from a single page"""
        try:
            headers = {"User-Agent": self.scraping_config["user_agent"]}

            async with session.get(url, headers=headers, timeout=self.scraping_config["request_timeout"]) as response:
                if response.status == 200:
                    html = await response.text()

                    soup = BeautifulSoup(html, 'html.parser')

                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()

                    # Get text content
                    text = soup.get_text()

                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)

                    # Limit content length
                    if len(text) > self.scraping_config["max_content_length"]:
                        text = text[:self.scraping_config["max_content_length"]] + "..."

                    return text

        except Exception as e:
            self.logger.debug(f"Error scraping page {url}: {e}")
            return None

    async def get_page_content(self, url: str) -> Optional[str]:
        """Get content from a specific URL"""
        try:
            async with aiohttp.ClientSession() as session:
                return await self._scrape_single_page(session, url)

        except Exception as e:
            self.logger.error(f"Error getting page content: {e}")
            return None

    async def search_github(self, query: str, language: str = "python", max_results: int = 10) -> List[Dict[str, Any]]:
        """Search GitHub repositories"""
        try:
            # GitHub search API
            api_url = f"https://api.github.com/search/repositories?q={query}+language:{language}&sort=stars&order=desc&per_page={max_results}"

            headers = {"Accept": "application/vnd.github.v3+json"}

            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()

                        results = []
                        for repo in data.get("items", []):
                            result = {
                                "title": repo.get("name", ""),
                                "url": repo.get("html_url", ""),
                                "description": repo.get("description", ""),
                                "stars": repo.get("stargazers_count", 0),
                                "language": repo.get("language", ""),
                                "source": "github",
                                "timestamp": time.time()
                            }
                            results.append(result)

                        return results
                    else:
                        self.logger.error(f"GitHub API error: {response.status}")
                        return []

        except Exception as e:
            self.logger.error(f"Error searching GitHub: {e}")
            return []

    async def search_stackoverflow(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Stack Overflow questions"""
        try:
            # Use Stack Exchange API
            api_url = "https://api.stackexchange.com/2.3/search"
            params = {
                "order": "desc",
                "sort": "relevance",
                "intitle": query,
                "site": "stackoverflow",
                "pagesize": max_results,
                "filter": "default"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        results = []
                        for question in data.get("items", []):
                            result = {
                                "title": question.get("title", ""),
                                "url": question.get("link", ""),
                                "score": question.get("score", 0),
                                "answer_count": question.get("answer_count", 0),
                                "source": "stackoverflow",
                                "timestamp": time.time()
                            }
                            results.append(result)

                        return results
                    else:
                        self.logger.error(f"Stack Overflow API error: {response.status}")
                        return []

        except Exception as e:
            self.logger.error(f"Error searching Stack Overflow: {e}")
            return []

    async def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search arXiv papers"""
        try:
            # arXiv API
            api_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params) as response:
                    if response.status == 200:
                        xml_data = await response.text()

                        # Parse XML (simplified)
                        soup = BeautifulSoup(xml_data, 'xml')

                        results = []
                        for entry in soup.find_all('entry')[:max_results]:
                            title = entry.find('title')
                            summary = entry.find('summary')
                            link = entry.find('id')

                            if title and link:
                                result = {
                                    "title": title.get_text() if title else "",
                                    "url": link.get_text() if link else "",
                                    "summary": summary.get_text() if summary else "",
                                    "source": "arxiv",
                                    "timestamp": time.time()
                                }
                                results.append(result)

                        return results
                    else:
                        self.logger.error(f"arXiv API error: {response.status}")
                        return []

        except Exception as e:
            self.logger.error(f"Error searching arXiv: {e}")
            return []

    def get_search_apis_status(self) -> Dict[str, Any]:
        """Get status of all search APIs"""
        status = {}

        for api_name, config in self.search_apis.items():
            status[api_name] = {
                "enabled": config["enabled"],
                "has_api_key": bool(config.get("api_key")),
                "base_url": config.get("base_url", "")
            }

        return status

    async def shutdown(self):
        """Shutdown web searcher"""
        try:
            self.logger.info("Shutting down web searcher...")

            # Clear cache
            self.search_cache.clear()

            self.logger.info("Web searcher shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down web searcher: {e}")