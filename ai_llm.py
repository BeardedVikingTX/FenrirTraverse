#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_llm.py – FenrirTraverse AI Intelligence Module
────────────────────────────────────────────────────
Handles all LLM interactions: key validation, token tracking,
prompt engineering, and response analysis with automatic provider
failover and comprehensive error handling.

Supports:
  - OpenRouter, OpenAI, Anthropic, Google Gemini, Mistral,
    DeepSeek (OpenAI‑compatible), Cohere, Hugging Face,
    Together AI, Groq, and local Ollama.
All calls are asynchronous to integrate seamlessly with the
async scanner core.

Author: BeardedViking
License: MIT
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from urllib.parse import urljoin

import aiohttp
import aiohttp.client_exceptions
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

# ─── Logging Setup ───────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─── Provider Definitions ────────────────────────────────────

class AIProvider(Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    TOGETHER = "together"
    GROQ = "groq"
    OLLAMA = "ollama"

@dataclass
class AIProviderConfig:
    """Configuration for a single LLM provider."""
    name: AIProvider
    api_key_env_var: str
    base_url: Optional[str] = None
    default_model: str = "gpt-3.5-turbo"
    token_limit: int = 100000
    timeout: int = 30
    max_retries: int = 3

# ─── Provider Registry ──────────────────────────────────────

PROVIDER_REGISTRY: Dict[AIProvider, AIProviderConfig] = {
    AIProvider.OPENROUTER: AIProviderConfig(
        name=AIProvider.OPENROUTER,
        api_key_env_var="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-4o-mini",
        token_limit=200000,
    ),
    AIProvider.OPENAI: AIProviderConfig(
        name=AIProvider.OPENAI,
        api_key_env_var="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        token_limit=100000,
    ),
    AIProvider.ANTHROPIC: AIProviderConfig(
        name=AIProvider.ANTHROPIC,
        api_key_env_var="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com/v1",
        default_model="claude-3-haiku-20240307",
        token_limit=100000,
    ),
    AIProvider.GOOGLE: AIProviderConfig(
        name=AIProvider.GOOGLE,
        api_key_env_var="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model="gemini-1.5-flash",
        token_limit=100000,
    ),
    AIProvider.MISTRAL: AIProviderConfig(
        name=AIProvider.MISTRAL,
        api_key_env_var="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-small-latest",
        token_limit=100000,
    ),
    AIProvider.DEEPSEEK: AIProviderConfig(
        name=AIProvider.DEEPSEEK,
        api_key_env_var="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        token_limit=100000,
    ),
    AIProvider.COHERE: AIProviderConfig(
        name=AIProvider.COHERE,
        api_key_env_var="COHERE_API_KEY",
        base_url="https://api.cohere.ai/v1",
        default_model="command-r",
        token_limit=100000,
    ),
    AIProvider.HUGGINGFACE: AIProviderConfig(
        name=AIProvider.HUGGINGFACE,
        api_key_env_var="HUGGINGFACE_API_KEY",
        base_url="https://api-inference.huggingface.co/models",
        default_model="meta-llama/Llama-3.2-3B-Instruct",
        token_limit=50000,
    ),
    AIProvider.TOGETHER: AIProviderConfig(
        name=AIProvider.TOGETHER,
        api_key_env_var="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-3.2-3B-Instruct",
        token_limit=100000,
    ),
    AIProvider.GROQ: AIProviderConfig(
        name=AIProvider.GROQ,
        api_key_env_var="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        default_model="mixtral-8x7b-32768",
        token_limit=100000,
    ),
    AIProvider.OLLAMA: AIProviderConfig(
        name=AIProvider.OLLAMA,
        api_key_env_var="OLLAMA_HOST",
        base_url="http://localhost:11434",
        default_model="llama3.2",
        token_limit=1000000,  # effectively unlimited locally
    ),
}

# ─── Token Manager ──────────────────────────────────────────

class TokenManager:
    """
    Tracks token usage per provider and enforces per‑provider
    token limits. Used to avoid exhausting paid quotas unexpectedly.
    """
    def __init__(self):
        self._usage: Dict[str, int] = {}
        self._limits: Dict[str, int] = {}

    def set_limit(self, provider: AIProvider, limit: int) -> None:
        self._limits[provider.value] = limit

    def record_usage(self, provider: AIProvider, tokens: int) -> None:
        key = provider.value
        self._usage[key] = self._usage.get(key, 0) + tokens

    def get_remaining(self, provider: AIProvider) -> int:
        key = provider.value
        used = self._usage.get(key, 0)
        limit = self._limits.get(key, 100000)
        return max(0, limit - used)

    def is_exhausted(self, provider: AIProvider) -> bool:
        return self.get_remaining(provider) <= 0

    def reset(self) -> None:
        self._usage.clear()

# ─── Main AI Engine ──────────────────────────────────────────

class AIEngine:
    """
    Asynchronous AI intelligence engine. Discovers available
    providers from environment variables, manages token budgets,
    and performs intelligent response analysis with automatic
    failover.
    """

    def __init__(
        self,
        provider_order: Optional[List[str]] = None,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
    ):
        """
        :param provider_order: List of provider names (strings) in priority order.
                               Default: ["openrouter", "openai", "anthropic", ...]
        :param cache_enabled: If True, cache analysis results for identical content.
        :param cache_ttl: Time‑to‑live for cache entries in seconds.
        """
        self.token_manager = TokenManager()
        self.provider_order = provider_order or [
            "openrouter", "openai", "anthropic", "mistral",
            "deepseek", "google", "cohere", "huggingface",
            "together", "groq", "ollama",
        ]
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, dict]] = {}  # content_hash -> (timestamp, result)

        # Auto‑discover available providers
        self.available_providers: List[AIProvider] = []
        self._init_available_providers()

        # Set token limits from config
        for provider, config in PROVIDER_REGISTRY.items():
            self.token_manager.set_limit(provider, config.token_limit)

        logger.info(f"AI Engine initialized. Available providers: {[p.value for p in self.available_providers]}")

    # ─── Provider Discovery ───────────────────────────────────

    def _init_available_providers(self) -> None:
        """Scan environment variables to determine which providers are configured."""
        for provider, config in PROVIDER_REGISTRY.items():
            key = os.getenv(config.api_key_env_var)
            if provider == AIProvider.OLLAMA:
                # Ollama uses a host env var; if set, we consider it available
                if os.getenv(config.api_key_env_var):
                    self.available_providers.append(provider)
            else:
                if key and key.strip() and not key.startswith("your_"):
                    self.available_providers.append(provider)

    # ─── Provider Selection ──────────────────────────────────

    def get_best_provider(self) -> Optional[AIProvider]:
        """
        Return the first provider in the priority list that is available
        and has remaining tokens.
        """
        for provider_name in self.provider_order:
            try:
                provider = AIProvider(provider_name)
                if provider in self.available_providers:
                    if not self.token_manager.is_exhausted(provider):
                        return provider
            except ValueError:
                continue
        return None

    # ─── Core Analysis Entry Point ──────────────────────────

    async def analyze_response(
        self,
        url: str,
        status: int,
        content: str,
        headers: Dict[str, str],
        max_content_len: int = 8000,
    ) -> Dict[str, Any]:
        """
        Analyze an HTTP response using the best available LLM.

        :param url: Requested URL.
        :param status: HTTP status code.
        :param content: Response body (up to max_content_len).
        :param headers: Response headers.
        :param max_content_len: Truncate content to this length.
        :return: Structured analysis dict with keys:
                 - contains_secrets: bool
                 - severity: "Critical" | "High" | "Medium" | "Low" | "None"
                 - findings: list of strings
                 - summary: str
                 - recommendation: str
                 - raw: original full response (if parsing fails)
        """
        # ── Cache check ──
        if self.cache_enabled:
            cache_key = self._compute_cache_key(content, headers)
            if cache_key in self._cache:
                timestamp, cached_result = self._cache[cache_key]
                if asyncio.get_event_loop().time() - timestamp < self.cache_ttl:
                    logger.debug("Returning cached analysis for %s", url)
                    return cached_result
                else:
                    del self._cache[cache_key]  # expired

        # ── Find provider ──
        provider = self.get_best_provider()
        if not provider:
            logger.warning("No AI provider available for analysis.")
            return {"error": "No AI provider available", "contains_secrets": False}

        logger.info("Using AI provider: %s for %s", provider.value, url)

        # ── Build prompt ──
        prompt = self._build_analysis_prompt(url, status, content[:max_content_len], headers)

        # ── Call provider with retries ──
        try:
            raw_response = await self._call_provider_with_retry(provider, prompt)
            tokens_used = self._estimate_tokens(prompt + raw_response)
            self.token_manager.record_usage(provider, tokens_used)
            result = self._parse_analysis(raw_response)
            result["_provider"] = provider.value
            result["_tokens_used"] = tokens_used

            # Cache result
            if self.cache_enabled:
                self._cache[cache_key] = (asyncio.get_event_loop().time(), result)

            return result

        except Exception as e:
            logger.error("Analysis failed for %s: %s", url, str(e), exc_info=True)
            return {"error": str(e), "contains_secrets": False, "severity": "Unknown"}

    # ─── Prompt Engineering ──────────────────────────────────

    @staticmethod
    def _build_analysis_prompt(url: str, status: int, content: str, headers: dict) -> str:
        """
        Build a detailed analysis prompt. Designed to be concise
        yet comprehensive, instructing the model to return valid JSON.
        """
        # Truncate headers to avoid token waste
        headers_str = json.dumps(headers, indent=2, ensure_ascii=False)[:500]

        return f"""You are Fenrir, an expert security AI. Analyze the following HTTP response 
for sensitive information that could indicate a security vulnerability or misconfiguration.

URL: {url}
HTTP Status: {status}
Headers (truncated):
{headers_str}

Response Content (first {len(content)} chars):
{content}

Analyze for:
1. **Secrets**: API keys, passwords, tokens, database credentials, JWT secrets.
2. **PII**: Emails, phone numbers, addresses, SSNs, credit card numbers.
3. **Config data**: Database names, internal IPs, server paths, cloud instance IDs.
4. **Source code**: PHP, Python, JavaScript, or configuration file contents.
5. **Sensitive file indicators**: '.env', '.git', '.aws', 'id_rsa', etc.

Return your analysis **only** as a JSON object with these keys:
- "contains_secrets": boolean (true if any sensitive data is present)
- "severity": string: "Critical", "High", "Medium", "Low", or "None"
- "findings": list of strings, each describing a specific finding with quotes
- "summary": string, one‑sentence overview
- "recommendation": string, actionable advice

Example output:
{{
  "contains_secrets": true,
  "severity": "High",
  "findings": [
    "Found AWS secret key: AKIA...",
    "Discovered database password in configuration block"
  ],
  "summary": "AWS credentials and database password exposed in plaintext.",
  "recommendation": "Remove sensitive values, use environment variables, rotate keys immediately."
}}

Do NOT include any extra text, code fences, or commentary. Only output the JSON.
"""

    # ─── Provider‑specific Calls (Async) ────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError)
        ),
        reraise=True,
    )
    async def _call_provider_with_retry(self, provider: AIProvider, prompt: str) -> str:
        """Wrapper that applies retry logic to the actual provider call."""
        config = PROVIDER_REGISTRY[provider]
        if provider == AIProvider.OPENROUTER:
            return await self._call_openrouter(config, prompt)
        elif provider == AIProvider.OPENAI:
            return await self._call_openai(config, prompt)
        elif provider == AIProvider.ANTHROPIC:
            return await self._call_anthropic(config, prompt)
        elif provider == AIProvider.GOOGLE:
            return await self._call_google(config, prompt)
        elif provider == AIProvider.MISTRAL:
            return await self._call_mistral(config, prompt)
        elif provider == AIProvider.DEEPSEEK:
            return await self._call_deepseek(config, prompt)
        elif provider == AIProvider.COHERE:
            return await self._call_cohere(config, prompt)
        elif provider == AIProvider.HUGGINGFACE:
            return await self._call_huggingface(config, prompt)
        elif provider == AIProvider.TOGETHER:
            return await self._call_together(config, prompt)
        elif provider == AIProvider.GROQ:
            return await self._call_groq(config, prompt)
        elif provider == AIProvider.OLLAMA:
            return await self._call_ollama(config, prompt)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    # ─── OpenAI (and DeepSeek) ──────────────────────────────

    @staticmethod
    async def _call_openai(config: AIProviderConfig, prompt: str) -> str:
        """Call OpenAI's Chat Completion API (async)."""
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError(f"Missing API key for {config.name.value}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1500,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                urljoin(config.base_url, "/chat/completions"),
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"OpenAI API error {resp.status}: {error_text}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    # ─── OpenRouter ──────────────────────────────────────────

    @staticmethod
    async def _call_openrouter(config: AIProviderConfig, prompt: str) -> str:
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError("Missing OpenRouter API key")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1500,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                urljoin(config.base_url, "/chat/completions"),
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"OpenRouter error {resp.status}: {error_text}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    # ─── Anthropic ───────────────────────────────────────────

    @staticmethod
    async def _call_anthropic(config: AIProviderConfig, prompt: str) -> str:
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError("Missing Anthropic API key")

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
            "temperature": 0.2,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                urljoin(config.base_url, "/messages"),
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Anthropic error {resp.status}: {error_text}")
                data = await resp.json()
                return data["content"][0]["text"]

    # ─── Google Gemini ───────────────────────────────────────

    @staticmethod
    async def _call_google(config: AIProviderConfig, prompt: str) -> str:
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError("Missing Google API key")

        url = f"{config.base_url}/models/{config.default_model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1500},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Google Gemini error {resp.status}: {error_text}")
                data = await resp.json()
                # Extract text from response
                candidates = data.get("candidates", [])
                if candidates:
                    return candidates[0]["content"]["parts"][0]["text"]
                else:
                    raise RuntimeError("No response from Gemini")

    # ─── Mistral ─────────────────────────────────────────────

    @staticmethod
    async def _call_mistral(config: AIProviderConfig, prompt: str) -> str:
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError("Missing Mistral API key")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1500,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                urljoin(config.base_url, "/chat/completions"),
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Mistral error {resp.status}: {error_text}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    # ─── DeepSeek (OpenAI‑compatible) ──────────────────────

    async def _call_deepseek(self, config: AIProviderConfig, prompt: str) -> str:
        # DeepSeek uses OpenAI‑compatible endpoint; reuse OpenAI logic with different base URL
        # But we override base_url from config
        return await self._call_openai(config, prompt)

    # ─── Cohere ──────────────────────────────────────────────

    @staticmethod
    async def _call_cohere(config: AIProviderConfig, prompt: str) -> str:
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError("Missing Cohere API key")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.default_model,
            "message": prompt,
            "temperature": 0.2,
            "max_tokens": 1500,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                urljoin(config.base_url, "/chat"),
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Cohere error {resp.status}: {error_text}")
                data = await resp.json()
                return data["text"]

    # ─── Hugging Face Inference ─────────────────────────────

    @staticmethod
    async def _call_huggingface(config: AIProviderConfig, prompt: str) -> str:
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError("Missing Hugging Face API key")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1500,
                "temperature": 0.2,
                "return_full_text": False,
            },
        }
        # HF API uses the model name as part of the URL
        model_id = config.default_model
        url = f"{config.base_url}/{model_id}"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Hugging Face error {resp.status}: {error_text}")
                data = await resp.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get("generated_text", "")
                elif isinstance(data, dict):
                    return data.get("generated_text", "")
                else:
                    raise RuntimeError("Unexpected response format from Hugging Face")

    # ─── Together AI ─────────────────────────────────────────

    @staticmethod
    async def _call_together(config: AIProviderConfig, prompt: str) -> str:
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError("Missing Together AI API key")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1500,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                urljoin(config.base_url, "/chat/completions"),
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Together AI error {resp.status}: {error_text}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    # ─── Groq ─────────────────────────────────────────────────

    @staticmethod
    async def _call_groq(config: AIProviderConfig, prompt: str) -> str:
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError("Missing Groq API key")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1500,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                urljoin(config.base_url, "/chat/completions"),
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Groq error {resp.status}: {error_text}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    # ─── Ollama (local) ──────────────────────────────────────

    @staticmethod
    async def _call_ollama(config: AIProviderConfig, prompt: str) -> str:
        """
        Calls Ollama's /api/generate endpoint.
        Expects OLLAMA_HOST to point to the server (e.g., http://localhost:11434).
        """
        host = os.getenv(config.api_key_env_var, "http://localhost:11434")
        url = urljoin(host, "/api/generate")
        payload = {
            "model": config.default_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Ollama error {resp.status}: {error_text}")
                data = await resp.json()
                return data.get("response", "")

    # ─── Token Estimation ────────────────────────────────────

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Approximate token count using a simple character‑based rule.
        More accurate than 4 chars/token for English; use tiktoken if installed.
        """
        # Try to use tiktoken if available for OpenAI models
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: 4 chars ≈ 1 token (rough)
            return len(text) // 4

    # ─── Response Parsing ────────────────────────────────────

    @staticmethod
    def _parse_analysis(raw: str) -> Dict[str, Any]:
        """
        Extract JSON from the LLM response. If no JSON is found,
        return a structured error object.
        """
        # Look for JSON block between curly braces
        json_pattern = re.compile(r'\{[^{}]*\}', re.DOTALL)
        matches = json_pattern.findall(raw)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # Fallback: try to find a JSON-like structure
        start = raw.find('{')
        end = raw.rfind('}') + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass

        # If all fails, return a default with the raw text
        return {
            "contains_secrets": False,
            "severity": "None",
            "findings": [],
            "summary": "Unable to parse LLM response.",
            "recommendation": "Review raw response manually.",
            "raw_response": raw,
            "parse_error": True,
        }

    # ─── Caching Helper ──────────────────────────────────────

    @staticmethod
    def _compute_cache_key(content: str, headers: Dict[str, str]) -> str:
        """
        Compute a hash key based on the response content and relevant
        headers (e.g., Content‑Type) to avoid re‑analysing identical responses.
        """
        import hashlib
        data = content.encode('utf-8') + str(headers.get('content-type', '')).encode()
        return hashlib.sha256(data).hexdigest()

    # ─── Key Testing ──────────────────────────────────────────

    @staticmethod
    async def _test_key_with_lightweight_call(provider: AIProvider, config: AIProviderConfig) -> bool:
        """
        Perform a minimal API call to validate the key.
        Not all providers have a simple test endpoint; we use a cheap model.
        """
        try:
            # Use a tiny prompt
            prompt = "Hello"
            # We'll call the provider with a low token limit
            # Temporarily lower timeout
            original_timeout = config.timeout
            config.timeout = 10
            try:
                # Use the same _call_provider_with_retry but we need to avoid recursion
                # We'll directly call the provider method
                provider_method = getattr(AIEngine, f"_call_{provider.value}", None)
                if provider_method:
                    result = await provider_method(config, prompt)
                    return bool(result)
                return False
            finally:
                config.timeout = original_timeout
        except Exception:
            return False

    async def test_all_keys(self) -> Dict[str, bool]:
        """
        Test all configured API keys by making lightweight calls.
        Returns a dict {provider_name: is_valid}.
        """
        results = {}
        for provider, config in PROVIDER_REGISTRY.items():
            key = os.getenv(config.api_key_env_var)
            if key and key.strip() and not key.startswith("your_"):
                valid = await self._test_key_with_lightweight_call(provider, config)
                results[provider.value] = valid
            else:
                results[provider.value] = False
        return results

# ─── Factory Function ────────────────────────────────────────

def create_ai_engine(**kwargs) -> AIEngine:
    """Convenience factory for creating an AIEngine instance."""
    return AIEngine(**kwargs)

# ─── Example Usage (Async) ──────────────────────────────────

if __name__ == "__main__":
    # This is a self‑test to verify provider discovery and a simple analysis.
    async def main():
        engine = AIEngine()
        print("Available providers:", [p.value for p in engine.available_providers])

        # Test key validity
        key_status = await engine.test_all_keys()
        print("Key status:", json.dumps(key_status, indent=2))

        # Simulate a response analysis
        sample_url = "https://example.com/.env"
        sample_content = "DB_PASSWORD=supersecret\nAWS_ACCESS_KEY=AKIA123456\n"
        sample_headers = {"content-type": "text/plain"}

        result = await engine.analyze_response(sample_url, 200, sample_content, sample_headers)
        print("Analysis result:", json.dumps(result, indent=2))

    asyncio.run(main())
