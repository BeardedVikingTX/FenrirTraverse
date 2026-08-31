#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FenrirScanner – Deep‑Link Path Traversal Scanner
─────────────────────────────────────────────────
A lean, mean, verbose machine.
"""

import asyncio
import aiohttp
import argparse
import sys
import os
from pathlib import Path
from urllib.parse import urljoin
from datetime import datetime
import time

# Optional: use tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Install tqdm for a progress bar: pip install tqdm")

# ─── Banner ──────────────────────────────────────────────────
BANNER = r"""
  _____ _   _ _    _  ______                           _     
 |  ___| \ | (_)  | | | ___ \                         | |    
 | |__ |  \| |_  _| |_| |_/ / __ _ _ __ __ _  ___  __| |___ 
 |  __|| . ` | |/ _` |    / / _` | '__/ _` |/ _ \/ _` / __|
 | |___| |\  | | (_| | |\ \ (_| | | | (_| |  __/ (_| \__ \
 \____/\_| \_/_|\__,_|_| \_\__,_|_|  \__, |\___|\__,_|___/
                                       __/ |                
                                      |___/                 
"""

def print_banner():
    print(BANNER)
    print("  Deep‑Link Path Traversal Scanner")
    print("  =================================")
    print()

# ─── Core Logic ─────────────────────────────────────────────

async def test_one(session, url, sem, progress, output_dir, verbose_level):
    async with sem:
        try:
            async with session.get(url, timeout=10, allow_redirects=True) as resp:
                content = await resp.text(errors='ignore')
                if verbose_level >= 3:
                    # Show status for every request
                    status_str = f"{resp.status}"
                    if resp.status == 200:
                        status_str = f"\033[92m{status_str}\033[0m"  # green
                    elif resp.status in (301, 302, 307):
                        status_str = f"\033[94m{status_str}\033[0m"  # blue
                    elif resp.status in (403, 401):
                        status_str = f"\033[93m{status_str}\033[0m"  # yellow
                    elif resp.status in (404, 500):
                        status_str = f"\033[91m{status_str}\033[0m"  # red
                    print(f"{status_str}  {url}")
                if progress:
                    progress.update(1)

                # Check for hit
                if resp.status == 200 and len(content) > 0:
                    # Quick false‑positive filter
                    error_indicators = ["404", "not found", "access denied", "forbidden", "error"]
                    if not any(ind in content.lower() for ind in error_indicators):
                        # Save hit
                        domain = url.split('/')[2]  # extract domain
                        safe_fname = url.replace('/', '_').replace(':', '_')[:80]
                        out_file = Path(output_dir) / f"{domain}_{safe_fname}.txt"
                        out_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(out_file, 'w') as f:
                            f.write(f"# URL: {url}\n")
                            f.write(f"# Status: {resp.status}\n")
                            f.write(f"# Headers: {dict(resp.headers)}\n")
                            f.write("\n--- RESPONSE ---\n")
                            f.write(content)
                        print(f"\n\033[92m✓ HIT:\033[0m {url} ({len(content)} bytes) -> saved to {out_file}")
                        return True
                return False
        except asyncio.TimeoutError:
            if verbose_level >= 4:
                print(f"\033[90mTIMEOUT:\033[0m {url}")
            if progress:
                progress.update(1)
        except Exception as e:
            if verbose_level >= 3:
                print(f"\033[91mERROR:\033[0m {url} - {e}")
            if progress:
                progress.update(1)
        return False

# ─── Main ────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Deep‑Link Path Traversal Scanner")
    parser.add_argument("-d", "--domain", required=True, help="Target domain (e.g., https://example.com)")
    parser.add_argument("-p", "--payloads", default="deep_link_payloads.txt", help="Payloads file")
    parser.add_argument("-f", "--files", default="files.txt", help="Files list")
    parser.add_argument("-o", "--output", default="hits", help="Output directory")
    parser.add_argument("-c", "--concurrency", type=int, default=50, help="Concurrent requests")
    parser.add_argument("-t", "--timeout", type=int, default=10, help="Request timeout (seconds)")
    parser.add_argument("-v", "--verbose", type=int, default=5, choices=range(0,6),
                        help="Verbosity level: 0=silent, 5=max (default: 5)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    parser.add_argument("--banner", action="store_true", help="Show banner")
    args = parser.parse_args()

    if args.banner:
        print_banner()

    # Normalise domain
    domain = args.domain
    if not domain.startswith(("http://", "https://")):
        domain = "https://" + domain
    if not domain.endswith("/"):
        domain += "/"

    # Load payloads
    try:
        with open(args.payloads) as f:
            payloads = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Payloads file not found: {args.payloads}")
        sys.exit(1)

    # Load files
    try:
        with open(args.files) as f:
            files = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Files file not found: {args.files}")
        sys.exit(1)

    # Build URLs
    urls = []
    for p in payloads:
        for fname in files:
            urls.append(urljoin(domain, p.replace("{FILE}", fname)))

    total = len(urls)
    print(f"🚀 Testing {total} URLs on {domain}")
    print(f"   Payloads: {len(payloads)}, Files: {len(files)}, Concurrency: {args.concurrency}")
    print(f"   Verbosity: {args.verbose} (0=silent, 5=full) | Output: {args.output}/")
    print("   Press Ctrl+C to stop.\n")

    # Ensure output directory exists
    Path(args.output).mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)
    progress = None
    if HAS_TQDM and not args.no_progress:
        progress = tqdm(total=total, desc="Progress", unit="req")
    elif args.verbose >= 1:
        print("Progress: (no tqdm installed, use --no-progress to suppress this message)")
        # We'll just show a simple counter in the loop

    async with aiohttp.ClientSession() as session:
        tasks = [
            test_one(session, url, sem, progress, args.output, args.verbose)
            for url in urls
        ]
        results = await asyncio.gather(*tasks)
        hits = sum(results)

    if progress:
        progress.close()

    print(f"\n✅ Done! {hits} hits found (saved in {args.output}/)")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
        sys.exit(0)
