#!/usr/bin/env python3
"""
simple_scanner.py – The Most Basic Scanner Ever
─────────────────────────────────────────────────
Usage: python simple_scanner.py -d https://example.com
"""

import requests
import argparse
import sys
from urllib.parse import urljoin
from datetime import datetime
import os
import time

def main():
    parser = argparse.ArgumentParser(description="Super simple path traversal scanner")
    parser.add_argument("-d", "--domain", required=True, help="Target domain")
    parser.add_argument("-p", "--payload", help="Single payload (optional)")
    parser.add_argument("-f", "--file", help="Single file (optional)")
    parser.add_argument("--payloads", default="deep_link_payloads.txt", help="Payloads file")
    parser.add_argument("--files", default="files.txt", help="Files list")
    parser.add_argument("-o", "--output", default="hits", help="Output directory")
    args = parser.parse_args()

    # Normalise domain
    domain = args.domain
    if not domain.startswith(("http://", "https://")):
        domain = "https://" + domain
    if not domain.endswith("/"):
        domain += "/"

    # Load payloads
    if args.payload:
        payloads = [args.payload]
    else:
        with open(args.payloads) as f:
            payloads = [line.strip() for line in f if line.strip()]

    # Load files
    if args.file:
        files = [args.file]
    else:
        with open(args.files) as f:
            files = [line.strip() for line in f if line.strip()]

    # Create output dir
    os.makedirs(args.output, exist_ok=True)

    total = len(payloads) * len(files)
    print(f"🚀 Testing {total} URLs on {domain}")
    print(f"   Payloads: {len(payloads)}, Files: {len(files)}")
    print("   Press Ctrl+C to stop.\n")

    hits = 0
    count = 0

    for payload in payloads:
        for fname in files:
            url = urljoin(domain, payload.replace("{FILE}", fname))
            count += 1
            print(f"[{count}/{total}] Testing: {url}", end=" ", flush=True)

            try:
                resp = requests.get(url, timeout=10, allow_redirects=True)
                status = resp.status_code
                content = resp.text

                print(f"→ {status}", flush=True)

                # Check for hit: 200, non-empty, not an error page
                if status == 200 and len(content) > 0:
                    error_indicators = ["404", "not found", "access denied", "forbidden", "error"]
                    if not any(ind in content.lower() for ind in error_indicators):
                        hits += 1
                        # Save it
                        safe_name = url.replace('/', '_').replace(':', '_')[:80]
                        out_file = os.path.join(args.output, f"{safe_name}.txt")
                        with open(out_file, 'w') as f:
                            f.write(f"# URL: {url}\n")
                            f.write(f"# Status: {status}\n")
                            f.write(f"# Headers: {dict(resp.headers)}\n")
                            f.write("\n--- RESPONSE ---\n")
                            f.write(content)
                        print(f"   ✓ HIT! Saved to {out_file}")
                time.sleep(0.05)  # tiny delay to be gentle

            except Exception as e:
                print(f"→ ERROR: {e}", flush=True)

    print(f"\n✅ Done! {hits} hits found (saved in {args.output}/)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
        sys.exit(0)
