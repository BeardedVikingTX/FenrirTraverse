# 🐺 FenrirTraverse – Deep‑Link Path Traversal Scanner

> *“Break the chains of path filters – uncover what’s hidden.”*  

FenrirTraverse is a high‑speed, asynchronous path traversal scanner built for bug bounty hunters and security professionals. It combines a curated list of encoded traversal payloads with a targeted set of sensitive file names, then intelligently analyses responses – all wrapped in a stunning Sci‑Fi Hacker terminal UI.

---

## ✨ Features

- **Massive payload coverage** – Unicode, double‑encoding, hex, and more.
- **Asynchronous engine** – powered by `aiohttp` + `asyncio` for lightning‑fast scans.
- **Smart response validation** – filters out false positives by analysing content and headers.
- **AI‑powered analysis** (optional) – detects secrets, config data, and PII using multiple LLM backends with automatic failover.
- **Immersive HUD** – glowing neon colours, animations, and a matrix‑inspired interface built with `Textual` & `Rich`.
- **Viking‑themed aesthetics** – runes, bold typography, and a dark atmospheric palette.

---

## 📦 Installation

```bash
git clone https://github.com/BeardedVikingTX/FenrirTraverse
cd fenrir-traverse
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
OPTIONAL: You can easily run;
```
sudo chmod +x tools_setup.sh
./tools_setup.sh
```
In order to install any additional tools & resources that would be helpful for this operation.

## 🚀 Usage
```
python fenrir_traverse.py
```
You will be prompted for:
* **Target domain** _(e.g., `example.com` – `https://` is added automatically)._
* **Path to payloads file** _(default: `deep_link_payloads.txt`)._
* **Path to files list** _(default: `files.txt`)._
* **Optional LLM integration** – _supply your API keys via environment variables or a `.env` file._

The tool will then:
1. Read all payloads and file names.
2. Construct every combination (payload + file).
3. Send concurrent GET requests (following redirects).
4. Validate responses (status 200, non‑empty, no error page patterns).
5. Save successful hits (URL, cURL command, response body) into a hits/ directory.
6. Display live updates with a colourful progress bar and hit notifications.

## 📂 File Structure
```
fenrir-traverse/
├── fenrir_traverse.py        # Main script
├── deep_link_payloads.txt    # Payloads with {FILE} placeholder
├── files.txt                 # Target file names
├── hits/                     # (created) Saved successful responses
├── requirements.txt          # Python dependencies
├── LICENSE.md                # MIT License
└── README.md                 # This file
```

## ⚙️ Requirements
See `requirements.txt` for a full list. The key libraries are:
* **aiohttp** – _asynchronous HTTP client._
* **asyncio** – _concurrency framework._
* **textual & rich** – _terminal UI & styling._
* **beautifulsoup4** – _response content analysis._
* **python-dotenv** – _load environment variables._
* **openai / openrouter** – _optional AI integration._

## 🤖 AI Integration (Optional)
To enable intelligent response analysis, set the following environment variables (or create a `.env` file):
```
OPENROUTER_API_KEY=sk-or-v1-...
MISTRAL_API_KEY=...
DEEPSEEK_API_KEY=...
HUGGINGFACE_API_KEY=hf_...
```
The tool will automatically try each provider in order and fall back if one is exhausted or unavailable.

# ⚠️ Important Disclaimer
***This tool is intended for authorised security testing only.***
Use it ***only*** on systems you own or have explicit written permission to test. Unauthorised access to computer systems is illegal and unethical. The author assumes no liability for any misuse or damage caused by this software.  
_By using FenrirTraverse, you agree to these terms._

# 🛡️ Contributing
Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request.

# 📄 License
MIT – see [LICENSE.md](LICENSE.md) for details.

# 🔮 Acknowledgements
* Inspired by the Norse wolf **Fenrir** – the beast that breaks all bonds.
* Built with ❤️ by BeardedViking.

