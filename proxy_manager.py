#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
proxy_manager.py – FenrirTraverse Enhanced Proxy Manager
───────────────────────────────────────────────────────────
Handles BOTH free and premium proxies with automatic detection,
priority rotation, failure tracking, and health scoring.

Features:
- Auto-detects premium proxies (those with username:password)
- Prioritizes premium proxies over free ones
- Tracks success rate per proxy (health score)
- Smart rotation – prefers healthier proxies
- Background re-validation of dead proxies
- Supports HTTP, HTTPS, SOCKS5 with/without auth

Author: BeardedViking
License: MIT
"""

import asyncio
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Set
from urllib.parse import urlparse

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# ─── Proxy Data Model ────────────────────────────────────────

@dataclass
class Proxy:
    """Represents a single proxy with health tracking."""
    protocol: str          # 'http', 'https', 'socks5'
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None

    # Health tracking
    success_count: int = 0
    failure_count: int = 0
    last_used: Optional[datetime] = None
    is_premium: bool = False  # auto-detected

    @property
    def url(self) -> str:
        """Proxy URL for aiohttp (e.g., http://user:pass@host:port)."""
        auth = f"{self.username}:{self.password}@" if self.username else ""
        return f"{self.protocol}://{auth}{self.host}:{self.port}"

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return (self.success_count / total) * 100

    @property
    def is_alive(self) -> bool:
        """Proxy is alive if it has a decent success rate."""
        # If never used, consider alive
        if self.success_count + self.failure_count == 0:
            return True
        return self.success_rate >= 50.0  # 50% threshold

    @classmethod
    def from_string(cls, proxy_str: str) -> Optional['Proxy']:
        """
        Parse a proxy string. Auto-detects premium (auth) vs free.
        Format: proto://[user:pass@]host:port
        """
        proxy_str = proxy_str.strip()
        if not proxy_str:
            return None

        # If no protocol specified, assume http
        if not re.match(r'^[a-z]+://', proxy_str):
            proxy_str = f"http://{proxy_str}"

        parsed = urlparse(proxy_str)
        if not parsed.hostname or not parsed.port:
            return None

        protocol = parsed.scheme.lower()
        if protocol not in ('http', 'https', 'socks5', 'socks5h'):
            logger.warning(f"Unsupported proxy protocol: {protocol}")
            return None

        username = parsed.username
        password = parsed.password
        is_premium = bool(username and password)  # premium if auth present

        return cls(
            protocol=protocol,
            host=parsed.hostname,
            port=parsed.port,
            username=username,
            password=password,
            is_premium=is_premium,
        )

    def mark_success(self) -> None:
        self.success_count += 1
        self.last_used = datetime.now()

    def mark_failure(self) -> None:
        self.failure_count += 1
        self.last_used = datetime.now()

# ─── Proxy Pool Manager ─────────────────────────────────────

class ProxyManager:
    """
    Manages a hybrid pool of free and premium proxies.
    Prioritizes premium proxies, falls back to free ones,
    and tracks health for intelligent rotation.
    """

    def __init__(
        self,
        proxy_file: str = "proxies.txt",
        max_failures: int = 5,           # failures before blacklisting
        min_success_rate: float = 30.0,   # success % below this = dead
        validation_url: str = "http://httpbin.org/ip",
        validation_timeout: int = 5,
        validation_interval: int = 300,  # seconds between re‑validations
        auto_reload: bool = True,
        prefer_premium: bool = True,     # use premium first
    ):
        """
        :param proxy_file: Path to file containing proxies (one per line).
        :param max_failures: Consecutive failures before marking dead.
        :param min_success_rate: Minimum success % to keep alive.
        :param validation_url: URL used to test proxy functionality.
        :param validation_timeout: Timeout per validation request.
        :param validation_interval: How often to re‑validate dead proxies.
        :param auto_reload: If True, periodically reload the file.
        :param prefer_premium: If True, use premium proxies before free.
        """
        self.proxy_file = proxy_file
        self.max_failures = max_failures
        self.min_success_rate = min_success_rate
        self.validation_url = validation_url
        self.validation_timeout = validation_timeout
        self.validation_interval = validation_interval
        self.auto_reload = auto_reload
        self.prefer_premium = prefer_premium

        # Internal state
        self._all_proxies: List[Proxy] = []
        self._available_premium: List[Proxy] = []
        self._available_free: List[Proxy] = []
        self._dead: Set[str] = set()  # proxy URLs that are dead
        self._last_validation: Dict[str, float] = {}

        self._lock = asyncio.Lock()
        self._premium_index = 0
        self._free_index = 0

        self._validation_task = None
        self._stop_event = asyncio.Event()

    # ─── Loading ─────────────────────────────────────────────

    async def load(self) -> None:
        """Load proxies from file. Auto-classifies as premium/free."""
        try:
            with open(self.proxy_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except FileNotFoundError:
            logger.warning(f"Proxy file '{self.proxy_file}' not found. No proxies loaded.")
            return

        proxies = []
        premium_count = 0
        free_count = 0

        for line in lines:
            p = Proxy.from_string(line)
            if p:
                proxies.append(p)
                if p.is_premium:
                    premium_count += 1
                else:
                    free_count += 1
            else:
                logger.debug(f"Skipping invalid proxy line: {line}")

        async with self._lock:
            self._all_proxies = proxies
            self._available_premium = [p for p in proxies if p.is_premium]
            self._available_free = [p for p in proxies if not p.is_premium]
            self._dead.clear()
            self._last_validation.clear()
            self._premium_index = 0
            self._free_index = 0

        logger.info(
            f"Loaded {len(proxies)} proxies "
            f"({premium_count} premium, {free_count} free) "
            f"from {self.proxy_file}"
        )

    # ─── Validation ──────────────────────────────────────────

    @staticmethod
    async def _validate_single(proxy: Proxy, validation_url: str, timeout: int) -> bool:
        """Test a single proxy by making a request to validation_url."""
        try:
            # For SOCKS proxies, we need a special connector
            if proxy.protocol.startswith('socks'):
                try:
                    import aiohttp_socks
                    connector = aiohttp_socks.SocksConnector.from_url(proxy.url)
                except ImportError:
                    logger.warning("aiohttp_socks not installed, skipping SOCKS validation")
                    return False
            else:
                connector = aiohttp.TCPConnector(ssl=False)

            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(
                    validation_url,
                    proxy=proxy.url if not proxy.protocol.startswith('socks') else None,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    ssl=False,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'origin' in data:
                            return True
                    return False
        except Exception as e:
            logger.debug(f"Proxy {proxy.url} validation failed: {e}")
            return False

    async def validate_all(self, concurrency: int = 10) -> Dict[str, bool]:
        """
        Validate all proxies in the pool concurrently.
        Returns dict {proxy_url: is_alive}.
        """
        async with self._lock:
            proxies_to_check = self._all_proxies.copy()

        semaphore = asyncio.Semaphore(concurrency)

        async def check(p: Proxy) -> Tuple[str, bool]:
            async with semaphore:
                alive = await self._validate_single(p, self.validation_url, self.validation_timeout)
                return p.url, alive

        tasks = [check(p) for p in proxies_to_check]
        results = await asyncio.gather(*tasks)

        # Update internal state
        alive_urls = {url for url, alive in results if alive}
        async with self._lock:
            # Rebuild available lists based on validation
            self._available_premium = [
                p for p in self._all_proxies
                if p.is_premium and p.url in alive_urls
            ]
            self._available_free = [
                p for p in self._all_proxies
                if not p.is_premium and p.url in alive_urls
            ]
            self._dead = {p.url for p in self._all_proxies if p.url not in alive_urls}
            for p in self._all_proxies:
                self._last_validation[p.url] = asyncio.get_event_loop().time()

        logger.info(
            f"Validation complete: "
            f"{len(self._available_premium)} premium, "
            f"{len(self._available_free)} free alive"
        )

        return {url: alive for url, alive in results}

    # ─── Smart Rotation ──────────────────────────────────────

    async def get_proxy(self, prefer_premium: bool = None) -> Optional[Proxy]:
        """
        Get the next available proxy using smart rotation.
        - Prefers premium proxies if available and prefer_premium=True.
        - Prefers proxies with higher success rates.
        - Falls back to free proxies when premium are exhausted.
        """
        if prefer_premium is None:
            prefer_premium = self.prefer_premium

        async with self._lock:
            # Try premium first if configured
            if prefer_premium and self._available_premium:
                proxy = self._get_best_proxy_from_list(self._available_premium)
                if proxy:
                    return proxy

            # Fall back to free proxies
            if self._available_free:
                proxy = self._get_best_proxy_from_list(self._available_free)
                if proxy:
                    return proxy

            # No proxies available
            return None

    def _get_best_proxy_from_list(self, proxy_list: List[Proxy]) -> Optional[Proxy]:
        """
        Get the best proxy from a list using weighted selection.
        Prefers higher success rates and lower failure counts.
        """
        if not proxy_list:
            return None

        # Calculate weights based on success rate (higher = better)
        weights = []
        for p in proxy_list:
            # Base weight: success rate, but ensure living proxies get a chance
            if p.success_count + p.failure_count == 0:
                weight = 50.0  # Untested = medium weight
            else:
                weight = p.success_rate
            weights.append(weight)

        # Normalize weights (avoid division by zero)
        total = sum(weights)
        if total == 0:
            # All weights zero? Pick randomly
            return random.choice(proxy_list)

        # Weighted random selection
        r = random.uniform(0, total)
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return proxy_list[i]

        return proxy_list[-1]  # Fallback

    # ─── Success/Failure Tracking ────────────────────────────

    async def mark_success(self, proxy: Proxy) -> None:
        """Record a successful request and update health."""
        async with self._lock:
            proxy.mark_success()
            # If proxy was dead and now succeeds, revive it
            if proxy.url in self._dead:
                self._dead.remove(proxy.url)
                if proxy.is_premium:
                    self._available_premium.append(proxy)
                else:
                    self._available_free.append(proxy)
                logger.info(f"Proxy {proxy.url} revived!")

    async def mark_failure(self, proxy: Proxy) -> None:
        """Record a failure and potentially blacklist the proxy."""
        async with self._lock:
            proxy.mark_failure()

            # Check if proxy should be blacklisted
            should_blacklist = (
                proxy.failure_count >= self.max_failures or
                proxy.success_rate < self.min_success_rate
            )

            if should_blacklist:
                self._dead.add(proxy.url)
                if proxy in self._available_premium:
                    self._available_premium.remove(proxy)
                if proxy in self._available_free:
                    self._available_free.remove(proxy)
                logger.info(f"Proxy {proxy.url} blacklisted "
                           f"(failures: {proxy.failure_count}, "
                           f"rate: {proxy.success_rate:.1f}%)")

    # ─── Statistics ──────────────────────────────────────────

    def stats(self) -> Dict[str, any]:
        """Return comprehensive pool statistics."""
        total = len(self._all_proxies)
        premium_alive = len(self._available_premium)
        free_alive = len(self._available_free)

        # Calculate average success rates
        all_alive = self._available_premium + self._available_free
        avg_rate = sum(p.success_rate for p in all_alive) / len(all_alive) if all_alive else 0

        return {
            "total": total,
            "premium_alive": premium_alive,
            "free_alive": free_alive,
            "total_alive": premium_alive + free_alive,
            "dead": len(self._dead),
            "avg_success_rate": round(avg_rate, 2),
            "premium_ratio": f"{premium_alive}/{total}",
            "free_ratio": f"{free_alive}/{total}",
        }

    async def get_stats_async(self) -> Dict[str, any]:
        """Async wrapper for stats (acquires lock)."""
        async with self._lock:
            return self.stats()

    # ─── Background Validation ──────────────────────────────

    async def _background_validator(self) -> None:
        """Periodically re‑validate dead proxies and reload the file."""
        while not self._stop_event.is_set():
            try:
                # Reload new proxies if file changed
                if self.auto_reload:
                    await self.load()

                # Re‑validate dead proxies that haven't been checked recently
                now = asyncio.get_event_loop().time()
                to_revalidate = []
                async with self._lock:
                    for proxy in self._all_proxies:
                        if proxy.url in self._dead:
                            last = self._last_validation.get(proxy.url, 0)
                            if now - last > self.validation_interval:
                                to_revalidate.append(proxy)

                if to_revalidate:
                    logger.debug(f"Re-validating {len(to_revalidate)} dead proxies")
                    sem = asyncio.Semaphore(3)
                    async def recheck(p: Proxy):
                        async with sem:
                            alive = await self._validate_single(p, self.validation_url, self.validation_timeout)
                            if alive:
                                async with self._lock:
                                    self._dead.discard(p.url)
                                    if p.is_premium:
                                        self._available_premium.append(p)
                                    else:
                                        self._available_free.append(p)
                                    logger.info(f"Proxy {p.url} revived in background")
                            self._last_validation[p.url] = now

                    await asyncio.gather(*[recheck(p) for p in to_revalidate])

                await asyncio.sleep(self.validation_interval)

            except Exception as e:
                logger.error(f"Background validator error: {e}")
                await asyncio.sleep(60)

    async def start_background_validation(self) -> None:
        """Start the background validation task."""
        if self._validation_task is None:
            self._stop_event.clear()
            self._validation_task = asyncio.create_task(self._background_validator())

    async def stop_background_validation(self) -> None:
        """Stop the background validation task."""
        if self._validation_task:
            self._stop_event.set()
            await self._validation_task
            self._validation_task = None

    # ─── Context Manager ─────────────────────────────────────

    async def __aenter__(self):
        await self.load()
        await self.start_background_validation()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop_background_validation()

# ─── Factory Function ──────────────────────────────────────

def create_proxy_manager(**kwargs) -> ProxyManager:
    """Convenience factory for creating a ProxyManager."""
    return ProxyManager(**kwargs)

# ─── Example Usage ─────────────────────────────────────────

async def main():
    pm = await create_proxy_manager(
        proxy_file="proxies.txt",
        max_failures=3,
        min_success_rate=30.0,
        prefer_premium=True,
    ).__aenter__()

    print("Proxy stats:", await pm.get_stats_async())

    # Get a proxy (premium preferred)
    proxy = await pm.get_proxy()
    if proxy:
        print(f"Using proxy: {proxy.url} (premium: {proxy.is_premium})")

        # Simulate using it
        await pm.mark_success(proxy)

    await pm.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
