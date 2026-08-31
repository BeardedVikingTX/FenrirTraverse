#!/usr/bin/env python3
import asyncio
import aiohttp
from urllib.parse import urljoin
from tqdm import tqdm
import sys

async def test_one(session, url, sem, progress):
    async with sem:
        try:
            async with session.get(url, timeout=5) as resp:
                status = resp.status
                content_len = len(await resp.text())
                progress.update(1)
                # Only print hits or interesting statuses
                if status == 200:
                    print(f"✓ 200: {url} ({content_len} bytes)")
                elif status in (301, 302, 307):
                    print(f"➜ {status}: {url}")
                elif status in (403, 404, 500):
                    print(f"✗ {status}: {url}")
                else:
                    print(f"  {status}: {url}")
        except asyncio.TimeoutError:
            progress.update(1)
            print(f"⏱ TIMEOUT: {url}")
        except Exception as e:
            progress.update(1)
            print(f"⚠ ERROR: {url} - {e}")

async def main():
    # --- CONFIG (use a small test set) ---
    # Use only first 5 payloads and first 10 files for quick test
    with open("deep_link_payloads.txt") as f:
        all_payloads = [line.strip() for line in f if line.strip()]
    payloads = all_payloads[:5]  # just 5

    with open("files.txt") as f:
        all_files = [line.strip() for line in f if line.strip()]
    files = all_files[:10]        # just 10

    domain = "https://dropbox.com/"  # ensure trailing slash

    urls = []
    for p in payloads:
        for fname in files:
            urls.append(urljoin(domain, p.replace("{FILE}", fname)))

    print(f"🚀 Testing {len(urls)} URLs (payloads: {len(payloads)}, files: {len(files)})")
    print("Press Ctrl+C to stop.\n")

    sem = asyncio.Semaphore(20)  # lower concurrency for less chance of blocking
    progress = tqdm(total=len(urls), desc="Progress", unit="req")

    async with aiohttp.ClientSession() as session:
        tasks = [test_one(session, url, sem, progress) for url in urls]
        await asyncio.gather(*tasks)

    progress.close()
    print("\n✅ Done!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
