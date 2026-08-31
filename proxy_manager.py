#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
proxy_manager.py – FenrirTraverse Enhanced Proxy Manager (v3)
───────────────────────────────────────────────────────────
Handles BOTH free and premium proxies with automatic detection,
priority rotation, failure tracking, health scoring, and persistence.

Features:
- Auto-detects premium proxies (auth present)
- Supports SOCKS4, SOCKS5, HTTP, HTTPS
- Prioritizes premium > proven free > untested
- Tracks success rate per proxy (health score)
- Smart rotation – prefers healthier proxies
- Background re-validation of dead proxies
- Preserves health data across reloads (merges)
- Optional persistent state (save/load)
- Configurable validation URL(s)

Author: BeardedViking
License: MIT
"""

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Set, Any
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger(__name__)

# ─── Proxy Data Model ────────────────────────────────────────

@dataclass
class Proxy:
    """Represents a single proxy with health tracking."""
    protocol: str          # 'http', 'https', 'socks4', 'socks5', 'socks5h'
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
        """Proxy URL (e.g., http://user:pass@host:port)."""
        auth = f"{self.username}:{self.password}@" if self.username else ""
        return f"{self.protocol}://{auth}{self.host}:{self.port}"

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return (self.success_count / total) * 100

    @property
    def is_alive(self) -> bool:
        """Proxy considered alive if never used or success_rate ≥ 50%."""
        if self.success_count + self.failure_count == 0:
            return True
        return self.success_rate >= 50.0

    @classmethod
    def from_string(cls, proxy_str: str) -> Optional['Proxy']:
        """Parse proxy string. Auto-detects premium (auth)."""
        proxy_str = proxy_str.strip()
        if not proxy_str:
            return None

        # Default to http if no protocol
        if not re.match(r'^[a-z]+://', proxy_str):
            proxy_str = f"http://{proxy_str}"

        parsed = urlparse(proxy_str)
        if not parsed.hostname or not parsed.port:
            return None

        protocol = parsed.scheme.lower()
        allowed = {'http', 'https', 'socks4', 'socks5', 'socks5h'}
        if protocol not in allowed:
            logger.warning(f"Unsupported proxy protocol: {protocol}")
            return None

        username = parsed.username
        password = parsed.password
        is_premium = bool(username and password)

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

    def to_dict(self) -> dict:
        """Convert to dict for persistence."""
        d = asdict(self)
        d['last_used'] = self.last_used.isoformat() if self.last_used else None
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'Proxy':
        """Restore from dict."""
        last_used = data.get('last_used')
        if last_used:
            data['last_used'] = datetime.fromisoformat(last_used)
        return cls(**data)


# ─── Proxy Pool Manager ─────────────────────────────────────

class ProxyManager:
    """
    Manages a hybrid pool of free and premium proxies.
    Supports HTTP, HTTPS, SOCKS4, SOCKS5.
    """

    def __init__(
        self,
        proxy_file: str = "proxies.txt",
        max_failures: int = 5,
        min_success_rate: float = 30.0,
        validation_urls: Optional[List[str]] = None,
        validation_timeout: int = 5,
        validation_interval: int = 300,
        auto_reload: bool = True,
        prefer_premium: bool = True,
        persistent_state_file: Optional[str] = "proxy_state.json",
    ):
        """
        :param proxy_file: Path to file containing proxies (one per line).
        :param max_failures: Consecutive failures before blacklisting.
        :param min_success_rate: Minimum success % to keep alive.
        :param validation_urls: List of URLs to test proxy (fallback order).
        :param validation_timeout: Timeout per validation request.
        :param validation_interval: Seconds between re‑validations.
        :param auto_reload: If True, periodically reload file (merge).
        :param prefer_premium: Use premium proxies first.
        :param persistent_state_file: Save/load health to this JSON file.
        """
        self.proxy_file = proxy_file
        self.max_failures = max_failures
        self.min_success_rate = min_success_rate
        self.validation_urls = validation_urls or [
            "http://httpbin.org/ip",
            "http://ip-api.com/json",
        ]
        self.validation_timeout = validation_timeout
        self.validation_interval = validation_interval
        self.auto_reload = auto_reload
        self.prefer_premium = prefer_premium
        self.persistent_state_file = persistent_state_file

        # Internal state
        self._all_proxies: Dict[str, Proxy] = {}  # key: proxy.url
        self._available_premium: List[Proxy] = []
        self._available_free: List[Proxy] = []
        self._dead: Set[str] = set()  # proxy URLs that are dead
        self._last_validation: Dict[str, float] = {}
        self._lock = asyncio.Lock()

        self._validation_task = None
        self._stop_event = asyncio.Event()

    # ─── Loading / Persistence ──────────────────────────────

    async def load(self, merge: bool = True) -> None:
        """Load proxies from file. If merge=True, preserve health for existing proxies."""
        try:
            with open(self.proxy_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except FileNotFoundError:
            logger.warning(f"Proxy file '{self.proxy_file}' not found.")
            return

        new_proxies = {}
        for line in lines:
            p = Proxy.from_string(line)
            if p:
                new_proxies[p.url] = p

        async with self._lock:
            if merge and self._all_proxies:
                # Merge: keep old health data for proxies that still exist
                for url, old_p in self._all_proxies.items():
                    if url in new_proxies:
                        new_proxies[url].success_count = old_p.success_count
                        new_proxies[url].failure_count = old_p.failure_count
                        new_proxies[url].last_used = old_p.last_used

            self._all_proxies = new_proxies

            # Mark dead only if proxy has been used and is failing
            self._dead.clear()
            for url, p in self._all_proxies.items():
                total_attempts = p.success_count + p.failure_count
                if total_attempts > 0:
                    if p.failure_count >= self.max_failures or p.success_rate < self.min_success_rate:
                        self._dead.add(url)
                # else: untested -> alive

            # Now rebuild available lists based on updated _dead
            self._rebuild_available_lists()

            logger.info(
                f"Loaded {len(new_proxies)} proxies "
                f"(premium: {sum(1 for p in new_proxies.values() if p.is_premium)}, "
                f"free: {sum(1 for p in new_proxies.values() if not p.is_premium)})"
            )

    async def save_state(self) -> None:
        """Save current health data to JSON file."""
        if not self.persistent_state_file:
            return
        data = [p.to_dict() for p in self._all_proxies.values()]
        try:
            with open(self.persistent_state_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved proxy state to {self.persistent_state_file}")
        except Exception as e:
            logger.error(f"Failed to save proxy state: {e}")

    async def load_state(self) -> None:
        """Load health data from JSON file and merge with current proxies."""
        if not self.persistent_state_file:
            return
        try:
            with open(self.persistent_state_file, 'r') as f:
                data = json.load(f)
            restored = [Proxy.from_dict(item) for item in data]
            async with self._lock:
                restored_map = {p.url: p for p in restored}
                # Merge into existing proxies
                for url, p in self._all_proxies.items():
                    if url in restored_map:
                        p.success_count = restored_map[url].success_count
                        p.failure_count = restored_map[url].failure_count
                        p.last_used = restored_map[url].last_used
                self._rebuild_available_lists()
                self._dead = {url for url, p in self._all_proxies.items()
                              if p.failure_count >= self.max_failures or p.success_rate < self.min_success_rate}
            logger.info(f"Loaded proxy state from {self.persistent_state_file}")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Failed to load proxy state: {e}")

    def _rebuild_available_lists(self) -> None:
        """Rebuild premium/free lists from _all_proxies, excluding dead."""
        self._available_premium = [
            p for p in self._all_proxies.values()
            if p.is_premium and p.url not in self._dead
        ]
        self._available_free = [
            p for p in self._all_proxies.values()
            if not p.is_premium and p.url not in self._dead
        ]

    # ─── Validation ──────────────────────────────────────────

    @staticmethod
    async def _validate_single(proxy: Proxy, validation_urls: List[str], timeout: int) -> bool:
        """Test a single proxy against a list of URLs (fallback)."""
        for url in validation_urls:
            try:
                connector = None
                if proxy.protocol.startswith('socks'):
                    try:
                        import aiohttp_socks
                        connector = aiohttp_socks.SocksConnector.from_url(proxy.url)
                    except ImportError:
                        logger.warning("aiohttp_socks not installed, cannot validate SOCKS proxy")
                        return False
                else:
                    connector = aiohttp.TCPConnector(ssl=False)

                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.get(
                        url,
                        proxy=proxy.url if not proxy.protocol.startswith('socks') else None,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                        ssl=False if proxy.protocol.startswith('socks') else None,
                    ) as resp:
                        if resp.status == 200:
                            # Check if response contains IP or origin
                            text = await resp.text()
                            if any(k in text.lower() for k in ['ip', 'origin']):
                                return True
            except Exception as e:
                logger.debug(f"Proxy {proxy.url} failed validation via {url}: {e}")
                continue
        return False

    async def validate_all(self, concurrency: int = 20) -> Dict[str, bool]:
        """Validate all proxies concurrently. Returns {url: alive}."""
        async with self._lock:
            proxies = list(self._all_proxies.values())

        semaphore = asyncio.Semaphore(concurrency)

        async def check(p: Proxy) -> Tuple[str, bool]:
            async with semaphore:
                alive = await self._validate_single(p, self.validation_urls, self.validation_timeout)
                return p.url, alive

        tasks = [check(p) for p in proxies]
        results = await asyncio.gather(*tasks)

        alive_urls = {url for url, alive in results if alive}
        async with self._lock:
            # Update dead set and lists
            self._dead = {p.url for p in proxies if p.url not in alive_urls}
            self._rebuild_available_lists()
            now = asyncio.get_event_loop().time()
            for p in proxies:
                self._last_validation[p.url] = now

        logger.info(f"Validation done: {len(alive_urls)} alive, {len(proxies)-len(alive_urls)} dead")
        return {url: alive for url, alive in results}

    # ─── Smart Rotation ──────────────────────────────────────

    async def get_proxy(self, prefer_premium: Optional[bool] = None) -> Optional[Proxy]:
        """
        Get next available proxy using smart rotation.
        Order: premium > proven free (high success) > untested free.
        """
        if prefer_premium is None:
            prefer_premium = self.prefer_premium

        async with self._lock:
            # Premium first
            if prefer_premium and self._available_premium:
                proxy = self._select_best(self._available_premium)
                if proxy:
                    return proxy

            # Free proxies: separate proven vs untested
            proven = [p for p in self._available_free if p.success_count + p.failure_count > 0]
            untested = [p for p in self._available_free if p.success_count + p.failure_count == 0]

            # Prefer proven proxies with high success rate
            if proven:
                proxy = self._select_best(proven)
                if proxy:
                    return proxy

            # Fall back to untested
            if untested:
                return random.choice(untested)

            return None

    def _select_best(self, proxy_list: List[Proxy]) -> Optional[Proxy]:
        """Weighted selection: prefer higher success rate, penalize failures."""
        if not proxy_list:
            return None

        weights = []
        for p in proxy_list:
            if p.success_count + p.failure_count == 0:
                weight = 1.0   # untested very low
            else:
                # Weight = success_rate, but boost if recent successes
                weight = p.success_rate
                # Add bonus for consecutive successes?
                if p.success_count > p.failure_count:
                    weight += 20
                weight = max(weight, 1.0)
            weights.append(weight)

        total = sum(weights)
        if total == 0:
            return random.choice(proxy_list)

        r = random.uniform(0, total)
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return proxy_list[i]
        return proxy_list[-1]

    # ─── Success/Failure Tracking ────────────────────────────

    async def mark_success(self, proxy: Proxy) -> None:
        async with self._lock:
            proxy.mark_success()
            if proxy.url in self._dead:
                self._dead.remove(proxy.url)
                self._rebuild_available_lists()
                logger.info(f"Proxy {proxy.url} revived!")

    async def mark_failure(self, proxy: Proxy) -> None:
        async with self._lock:
            proxy.mark_failure()
            if proxy.failure_count >= self.max_failures or proxy.success_rate < self.min_success_rate:
                self._dead.add(proxy.url)
                self._rebuild_available_lists()
                logger.info(f"Proxy {proxy.url} blacklisted "
                           f"(failures: {proxy.failure_count}, rate: {proxy.success_rate:.1f}%)")

    # ─── Statistics ──────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        total = len(self._all_proxies)
        premium_alive = len(self._available_premium)
        free_alive = len(self._available_free)
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

    async def get_stats_async(self) -> Dict[str, Any]:
        async with self._lock:
            return self.stats()

    # ─── Background Validation ──────────────────────────────

    async def _background_validator(self) -> None:
        """Periodically re‑validate dead proxies and reload file (merge)."""
        while not self._stop_event.is_set():
            try:
                if self.auto_reload:
                    await self.load(merge=True)

                now = asyncio.get_event_loop().time()
                to_revalidate = []
                async with self._lock:
                    for url in self._dead:
                        if now - self._last_validation.get(url, 0) > self.validation_interval:
                            proxy = self._all_proxies.get(url)
                            if proxy:
                                to_revalidate.append(proxy)

                if to_revalidate:
                    logger.debug(f"Re-validating {len(to_revalidate)} dead proxies")
                    sem = asyncio.Semaphore(5)
                    async def recheck(p: Proxy):
                        async with sem:
                            alive = await self._validate_single(p, self.validation_urls, self.validation_timeout)
                            if alive:
                                async with self._lock:
                                    self._dead.discard(p.url)
                                    p.failure_count = 0  # reset failures if revived
                                    self._rebuild_available_lists()
                                    logger.info(f"Proxy {p.url} revived in background")
                            self._last_validation[p.url] = now

                    await asyncio.gather(*[recheck(p) for p in to_revalidate])

                await asyncio.sleep(self.validation_interval)

            except Exception as e:
                logger.error(f"Background validator error: {e}")
                await asyncio.sleep(60)

    async def start_background_validation(self) -> None:
        if self._validation_task is None:
            self._stop_event.clear()
            self._validation_task = asyncio.create_task(self._background_validator())

    async def stop_background_validation(self) -> None:
        if self._validation_task:
            self._stop_event.set()
            await self._validation_task
            self._validation_task = None

    # ─── Context Manager ─────────────────────────────────────

    async def __aenter__(self):
        await self.load(merge=False)  # initial load
        if self.persistent_state_file:
            await self.load_state()   # load health if exists
        await self.start_background_validation()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop_background_validation()
        if self.persistent_state_file:
            await self.save_state()


# ─── Factory Function ──────────────────────────────────────

def create_proxy_manager(**kwargs) -> ProxyManager:
    return ProxyManager(**kwargs)


# ─── Example Usage ─────────────────────────────────────────

async def main():
    # Use a small file for quick testing
    pm = await create_proxy_manager(
        proxy_file="proxies.txt",   # <-- small file
        max_failures=3,
        min_success_rate=30.0,
        prefer_premium=True,
        persistent_state_file=None,        # disable persistence for clean test
        validation_timeout=3,              # 3 seconds
    ).__aenter__()

    total = len(pm._all_proxies)
    print(f"Loaded {total} proxies. Validating...")

    results = await pm.validate_all(concurrency=30)
    alive = sum(1 for v in results.values() if v)
    print(f"Alive: {alive}/{total}")

    print("Proxy stats:", await pm.get_stats_async())

    proxy = await pm.get_proxy()
    if proxy:
        print(f"Using proxy: {proxy.url} (premium: {proxy.is_premium})")
        await pm.mark_success(proxy)
    else:
        print("No working proxy found.")

    await pm.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())
