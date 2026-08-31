# ─── AI / LLM API Keys ─────────────────────────────────────
# OpenRouter (unified gateway)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OpenAI (GPT-4, GPT-3.5, etc.)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Google Gemini / PaLM
GOOGLE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Mistral AI
MISTRAL_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# DeepSeek (uses OpenAI-compatible endpoint, just set base URL)
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# Cohere
COHERE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Hugging Face
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Together AI
TOGETHER_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Replicate
REPLICATE_API_KEY=r8_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Groq
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Perplexity AI
PERPLEXITY_API_KEY=pplx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ─── Local / Self-Hosted ──────────────────────────────────
# Ollama (usually runs locally, no key needed)
OLLAMA_HOST=http://localhost:11434

# ─── Proxy / Network ──────────────────────────────────────
# HTTP/HTTPS proxy (optional)
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080

# Tor SOCKS proxy (optional)
TOR_PROXY=socks5h://127.0.0.1:9050

# ─── Scan Configuration ───────────────────────────────────
# Concurrency level (number of simultaneous requests)
CONCURRENCY_LIMIT=50

# Request timeout in seconds
REQUEST_TIMEOUT=10

# Maximum redirects to follow
MAX_REDIRECTS=5

# Rate limit (requests per second) – 0 = no limit
RATE_LIMIT=100

# User-Agent randomization (true/false)
RANDOM_USER_AGENT=true

# Enable/disable AI analysis (true/false)
AI_ANALYSIS_ENABLED=true

# Preferred LLM order (comma-separated) – will fall back in this order
AI_PROVIDER_ORDER=openrouter,openai,anthropic,mistral,deepseek,google,cohere,huggingface

# ─── Logging & Output ─────────────────────────────────────
# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Save all responses (even errors) for debugging
SAVE_ALL_RESPONSES=false

# Output directory for findings
OUTPUT_DIR=hits
