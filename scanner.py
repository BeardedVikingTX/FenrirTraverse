#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scanner.py – FenrirTraverse Asynchronous Path Traversal Scanner (Memory-Efficient)
─────────────────────────────────────────────────────────────
Combines deep‑link payloads with sensitive file names,
uses proxy rotation, optional AI‑powered response analysis,
and rotating user‑agents per request.

Usage:
    python scanner.py -d https://example.com [options]
"""

import asyncio
import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Generator
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientTimeout

# Import our modules
from proxy_manager import ProxyManager, create_proxy_manager
from ai_llm import AIEngine, create_ai_engine

# ─── Logging Setup ─────────────────────────────────────────
logger = logging.getLogger("fenrir_scanner")

def setup_logging(verbose_level: int):
    """Configure logging based on verbosity level (1-5)."""
    if verbose_level == 1:
        level = logging.WARNING
    elif verbose_level == 2:
        level = logging.INFO
    else:  # 3,4,5 all use DEBUG for now
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ─── Configuration dataclass ────────────────────────────────
@dataclass
class ScanConfig:
    domain: str
    payload_file: str
    files_file: str
    output_dir: str
    concurrency: int
    timeout: int
    max_redirects: int
    rate_limit: float
    proxy_file: Optional[str]
    use_proxies: bool
    ai_enabled: bool
    ai_provider_order: Optional[List[str]]
    max_retries: int
    save_all_responses: bool
    user_agent: str
    random_user_agent: bool
    user_agents_file: str
    verbose: int
    max_urls: int          # 0 = unlimited


# ─── Helper functions ───────────────────────────────────────
def load_lines(filename: str) -> List[str]:
    """Load non‑empty lines from a file."""
    try:
        with open(filename, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        sys.exit(1)


def load_user_agents(filename: str) -> List[str]:
    """Load user agents from file; if file missing, return empty list."""
    try:
        return load_lines(filename)
    except SystemExit:
        logger.warning(f"User agents file '{filename}' not found. Using fallback UA.")
        return []


def url_generator(domain: str, payloads: List[str], files: List[str], max_urls: int = 0) -> Generator[str, None, None]:
    """Yield URLs lazily. If max_urls > 0, stop after that many."""
    count = 0
    for payload in payloads:
        for file in files:
            path = payload.replace("{FILE}", file)
            yield urljoin(domain, path)
            count += 1
            if max_urls > 0 and count >= max_urls:
                return


def is_hit(status: int, content: str) -> bool:
    """Determine if response is a valid hit (200 and not an error page)."""
    if status != 200 or not content:
        return False
    error_indicators = ["404", "not found", "access denied", "forbidden", "error", "invalid"]
    lower = content.lower()
    if any(ind in lower for ind in error_indicators) and len(content) < 500:
        return False
    return True


# ─── Main scanner class ─────────────────────────────────────
class Scanner:
    def __init__(self, config: ScanConfig):
        self.config = config
        self.proxy_manager: Optional[ProxyManager] = None
        self.ai_engine: Optional[AIEngine] = None
        self.semaphore: asyncio.Semaphore = None
        self.session: aiohttp.ClientSession = None
        self.rate_limiter: RateLimiter = None
        self.hits = 0
        self.errors = 0
        self.total = 0
        self.start_time = time.time()
        self.completed_requests = 0
        self.user_agents: List[str] = []

        os.makedirs(config.output_dir, exist_ok=True)

    async def setup(self):
        """Initialize proxy manager, AI engine, user agents, and aiohttp session."""
        # Load user agents
        if self.config.random_user_agent and self.config.user_agents_file:
            self.user_agents = load_user_agents(self.config.user_agents_file)
            logger.info(f"Loaded {len(self.user_agents)} user agents for rotation")
        else:
            self.user_agents = [self.config.user_agent]
            logger.info("Using fixed user agent (random rotation disabled)")

        # Proxy manager
        if self.config.use_proxies and self.config.proxy_file:
            self.proxy_manager = await create_proxy_manager(
                proxy_file=self.config.proxy_file,
                max_failures=3,
                min_success_rate=30.0,
                prefer_premium=True,
                persistent_state_file="proxy_state.json" if self.config.use_proxies else None,
            ).__aenter__()
            logger.info(f"Proxy manager loaded {len(self.proxy_manager._all_proxies)} proxies")
        else:
            logger.info("Proxy rotation disabled")

        # AI engine
        if self.config.ai_enabled:
            self.ai_engine = create_ai_engine(provider_order=self.config.ai_provider_order)
            logger.info(f"AI engine initialized with providers: {[p.value for p in self.ai_engine.available_providers]}")
        else:
            logger.info("AI analysis disabled")

        # aiohttp session
        timeout = ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)

        # Concurrency control
        self.semaphore = asyncio.Semaphore(self.config.concurrency)

        # Rate limiter
        self.rate_limiter = RateLimiter(self.config.rate_limit)

    def get_random_user_agent(self) -> str:
        if self.user_agents:
            return random.choice(self.user_agents)
        return self.config.user_agent

    async def teardown(self):
        if self.session:
            await self.session.close()
        if self.proxy_manager:
            await self.proxy_manager.__aexit__(None, None, None)

    async def scan_url(self, url: str, retry_count: int = 0):
        async with self.semaphore:
            await self.rate_limiter.wait()

            proxy = None
            if self.proxy_manager:
                proxy = await self.proxy_manager.get_proxy()

            user_agent = self.get_random_user_agent()
            headers = {"User-Agent": user_agent}

            try:
                async with self.session.get(
                    url,
                    proxy=proxy.url if proxy else None,
                    allow_redirects=True,
                    ssl=False,
                    headers=headers,
                ) as resp:
                    status = resp.status
                    content = await resp.text()

                    if proxy:
                        if status >= 500:
                            await self.proxy_manager.mark_failure(proxy)
                        else:
                            await self.proxy_manager.mark_success(proxy)

                    if is_hit(status, content):
                        logger.info(f"✓ HIT: {url} (status {status})")
                        self.hits += 1

                        hit_data = {
                            "url": url,
                            "status": status,
                            "headers": dict(resp.headers),
                            "content": content,
                            "timestamp": time.time(),
                            "user_agent": user_agent,
                        }

                        if self.ai_engine:
                            try:
                                analysis = await self.ai_engine.analyze_response(
                                    url=url,
                                    status=status,
                                    content=content,
                                    headers=dict(resp.headers),
                                )
                                hit_data["ai_analysis"] = analysis
                                logger.info(f"   AI Analysis: {analysis.get('severity', 'Unknown')}")
                            except Exception as e:
                                logger.error(f"   AI analysis failed: {e}")

                        self.save_hit(hit_data)
                    else:
                        if self.config.save_all_responses:
                            self.save_response(url, status, content, resp.headers, user_agent)

                    self.completed_requests += 1
                    return status

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.debug(f"Connection error for {url}: {e}")
                if proxy:
                    await self.proxy_manager.mark_failure(proxy)
                self.errors += 1
                self.completed_requests += 1

                if retry_count < self.config.max_retries:
                    logger.debug(f"Retrying {url} (attempt {retry_count + 1})")
                    return await self.scan_url(url, retry_count + 1)
                return None

    def save_hit(self, hit_data: Dict[str, Any]):
        safe_name = hit_data["url"].replace("/", "_").replace(":", "_")[:100]
        out_file = os.path.join(self.config.output_dir, f"hit_{safe_name}.json")
        try:
            with open(out_file, "w") as f:
                json.dump(hit_data, f, indent=2, ensure_ascii=False)
            logger.info(f"   Saved to {out_file}")
        except Exception as e:
            logger.error(f"Failed to save hit: {e}")

    def save_response(self, url: str, status: int, content: str, headers: Dict, user_agent: str):
        safe_name = url.replace("/", "_").replace(":", "_")[:80]
        out_file = os.path.join(self.config.output_dir, f"resp_{safe_name}.txt")
        with open(out_file, "w") as f:
            f.write(f"URL: {url}\nStatus: {status}\nHeaders: {headers}\nUser-Agent: {user_agent}\n\n{content}")

    async def progress_reporter(self, stop_event: asyncio.Event):
        while not stop_event.is_set():
            await asyncio.sleep(5)
            elapsed = time.time() - self.start_time
            done = self.completed_requests
            total = self.total if self.total > 0 else "?"
            print(f"\r[Progress] {done}/{total} requests, Hits: {self.hits}, Errors: {self.errors}, Elapsed: {elapsed:.1f}s", end="", flush=True)
        print()

    async def worker(self, queue: asyncio.Queue):
        """Worker that pulls URLs from queue and scans them."""
        while True:
            url = await queue.get()
            if url is None:
                queue.task_done()
                break
            try:
                await self.scan_url(url)
            except Exception as e:
                logger.error(f"Unexpected error scanning {url}: {e}")
                self.errors += 1
            finally:
                queue.task_done()

    async def run(self, urls: Generator[str, None, None]):
        """Scan URLs from generator with bounded concurrency."""
        # Count total if max_urls is set, otherwise we don't know total for progress
        if self.config.max_urls > 0:
            self.total = self.config.max_urls
            logger.info(f"Starting scan of up to {self.total} URLs")
        else:
            self.total = 0
            logger.info("Starting scan (unlimited URLs)")

        logger.info(f"Concurrency: {self.config.concurrency}, Rate limit: {self.config.rate_limit}/s")
        if self.proxy_manager:
            logger.info(f"Proxy rotation enabled with {len(self.proxy_manager._all_proxies)} proxies")
        if self.ai_engine:
            logger.info("AI analysis enabled")
        if self.config.random_user_agent:
            logger.info(f"User-Agent rotation enabled with {len(self.user_agents)} agents")
        else:
            logger.info("User-Agent rotation disabled (using fixed UA)")

        # Progress reporter
        stop_event = asyncio.Event()
        reporter_task = asyncio.create_task(self.progress_reporter(stop_event))

        # Create queue and workers
        queue = asyncio.Queue(maxsize=self.config.concurrency * 2)
        workers = [asyncio.create_task(self.worker(queue)) for _ in range(self.config.concurrency)]

        # Feed URLs into queue
        producer_done = asyncio.Event()

        async def producer():
            try:
                for url in urls:
                    await queue.put(url)
                # Signal workers to exit
                for _ in workers:
                    await queue.put(None)
            except Exception as e:
                logger.error(f"Producer error: {e}")
            finally:
                producer_done.set()

        producer_task = asyncio.create_task(producer())

        # Wait for producer to finish
        await producer_task

        # Wait for queue to be fully processed
        await queue.join()

        # Cancel workers (they have already received None)
        for w in workers:
            w.cancel()

        # Stop progress reporter
        stop_event.set()
        await reporter_task

        logger.info(f"\n{'='*50}")
        logger.info(f"Scan completed in {time.time() - self.start_time:.2f}s")
        logger.info(f"Total requests: {self.completed_requests}")
        logger.info(f"Hits: {self.hits}")
        logger.info(f"Errors: {self.errors}")
        logger.info(f"Output directory: {self.config.output_dir}")


# ─── Rate limiter class ─────────────────────────────────────
class RateLimiter:
    def __init__(self, rate: float):
        self.rate = rate
        self.last = time.monotonic()
        self.lock = asyncio.Lock()

    async def wait(self):
        if self.rate <= 0:
            return
        async with self.lock:
            now = time.monotonic()
            delay = (1.0 / self.rate) - (now - self.last)
            if delay > 0:
                await asyncio.sleep(delay)
            self.last = time.monotonic()


# ─── Main entry point ───────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="FenrirTraverse Path Traversal Scanner")
    parser.add_argument("-d", "--domain", required=True, help="Target domain (e.g., https://example.com)")
    parser.add_argument("-p", "--payload-file", default="deep_link_payloads.txt", help="Payloads file")
    parser.add_argument("-f", "--files-file", default="files.txt", help="Files list")
    parser.add_argument("-o", "--output", default="hits", help="Output directory")
    parser.add_argument("-c", "--concurrency", type=int, default=50, help="Number of concurrent requests")
    parser.add_argument("-t", "--timeout", type=int, default=10, help="Request timeout in seconds")
    parser.add_argument("--max-redirects", type=int, default=5, help="Maximum redirects to follow")
    parser.add_argument("--rate-limit", type=float, default=0, help="Requests per second (0 = unlimited)")
    parser.add_argument("--proxy-file", default="proxies.txt", help="Proxy list file")
    parser.add_argument("-np", "--no-proxies", action="store_true", help="Disable proxy rotation (default: enabled)")
    parser.add_argument("-nai", "--no-ai", action="store_true", help="Disable AI analysis (default: enabled)")
    parser.add_argument("--ai-provider-order", help="Comma-separated provider order for AI (e.g., 'mistral,deepseek')")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per URL with different proxy")
    parser.add_argument("--save-all-responses", action="store_true", help="Save all responses (for debugging)")
    parser.add_argument("--user-agent", default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        help="Fallback user agent")
    parser.add_argument("--random-user-agent", action="store_true",
                        help="Rotate user agent randomly per request (uses user-agents.txt)")
    parser.add_argument("--user-agents-file", default="user-agents.txt",
                        help="File containing list of user agents for rotation")
    parser.add_argument("--max-urls", type=int, default=0,
                        help="Maximum number of URLs to test (0 = unlimited)")
    parser.add_argument("-v", "--verbose", type=int, choices=[1,2,3,4,5], default=2,
                        help="Verbosity level: 1=minimal, 2=normal, 3=debug, 4=detailed, 5=extreme")
    return parser.parse_args()


def main():
    print("Initializing FenrirTraverse scanner...")
    args = parse_args()
    print(f"Verbosity: {args.verbose}, Domain: {args.domain}")

    setup_logging(args.verbose)

    domain = args.domain
    if not domain.startswith(("http://", "https://")):
        domain = "https://" + domain
    if not domain.endswith("/"):
        domain += "/"

    print("Loading payloads...")
    payloads = load_lines(args.payload_file)
    print(f"Loaded {len(payloads)} payloads.")

    print("Loading files...")
    files = load_lines(args.files_file)
    print(f"Loaded {len(files)} files.")

    if not payloads or not files:
        logger.error("No payloads or files loaded. Check files.")
        sys.exit(1)

    total_possible = len(payloads) * len(files)
    print(f"Possible URLs: {total_possible}")

    if args.max_urls > 0:
        print(f"Limiting to first {args.max_urls} URLs.")

    # Create URL generator (lazy)
    urls_gen = url_generator(domain, payloads, files, args.max_urls)

    ai_provider_order = None
    if args.ai_provider_order:
        ai_provider_order = [s.strip() for s in args.ai_provider_order.split(",")]

    config = ScanConfig(
        domain=domain,
        payload_file=args.payload_file,
        files_file=args.files_file,
        output_dir=args.output,
        concurrency=args.concurrency,
        timeout=args.timeout,
        max_redirects=args.max_redirects,
        rate_limit=args.rate_limit,
        proxy_file=args.proxy_file,
        use_proxies=not args.no_proxies,
        ai_enabled=not args.no_ai,
        ai_provider_order=ai_provider_order,
        max_retries=args.max_retries,
        save_all_responses=args.save_all_responses,
        user_agent=args.user_agent,
        random_user_agent=args.random_user_agent,
        user_agents_file=args.user_agents_file,
        verbose=args.verbose,
        max_urls=args.max_urls,
    )

    scanner = Scanner(config)
    asyncio.run(_run_scanner(scanner, urls_gen))


async def _run_scanner(scanner: Scanner, urls_gen):
    await scanner.setup()
    try:
        await scanner.run(urls_gen)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    finally:
        await scanner.teardown()


if __name__ == "__main__":
    main()
