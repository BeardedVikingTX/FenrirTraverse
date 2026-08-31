#!/bin/bash
# ──────────────────────────────────────────────────────────────
# FenrirTraverse – External Tools Installer
# ──────────────────────────────────────────────────────────────

echo "🐺 Installing FenrirTraverse external dependencies..."

# Update package lists
sudo apt update

# Core networking tools
sudo apt install -y curl wget netcat-openbsd nmap masscan

# HTTP testing tools
sudo apt install -y httrack gobuster ffuf

# DNS & subdomain tools
sudo apt install -y dnsutils amass sublist3r

# Security tools
sudo apt install -y sqlmap nikto whatweb wpscan

# Web development tools (for parsing)
sudo apt install -y jq html2text

# Proxy tools
sudo apt install -y mitmproxy

# Install popular Python-based tools via pip
pip install crackmapexec
pip install impacket
pip install bloodhound

# Install Go-based tools (if Go is installed)
if command -v go &> /dev/null; then
    go install -v github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest
    go install -v github.com/projectdiscovery/httpx/cmd/httpx@latest
    go install -v github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest
fi

echo "✅ All external tools installed successfully!"
