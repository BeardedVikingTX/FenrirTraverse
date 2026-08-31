#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_llm.py – FenrirTraverse AI Intelligence Module (Enhanced)
─────────────────────────────────────────────────────────────────
Handles multiple LLM providers with credit/availability checking,
automatic fallback, token tracking, and intelligent response analysis.
Now with robust error handling and provider prioritisation.

Author: BeardedViking
License: MIT
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

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
    chat_endpoint: str = "/chat/completions"

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
        chat_endpoint="/messages",
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
        chat_endpoint="/chat",
    ),
    AIProvider.HUGGINGFACE: AIProviderConfig(
        name=AIProvider.HUGGINGFACE,
        api_key_env_var="HUGGINGFACE_API_KEY",
        base_url="https://api-inference.huggingface.co/models",
        default_model="meta-llama/Llama-3.2-3B-Instruct",
        token_limit=50000,
        chat_endpoint="/{model}",
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
        token_limit=1000000,
        chat_endpoint="/api/generate",
    ),
}

# ─── Token Manager ──────────────────────────────────────────

class TokenManager:
    """Track token usage per provider."""
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

# ─── Helper: Key Validation ─────────────────────────────────

def _is_placeholder_key(key: Optional[str]) -> bool:
    """Return True if the key looks like a placeholder (xxxx, your_, etc.)."""
    if not key or not key.strip():
        return True
    key_lower = key.lower().strip()
    if key_lower.startswith("your_"):
        return True
    if "xxxx" in key_lower:
        return True
    # Check if the key consists mostly of 'x' characters
    if len(key) > 4 and all(c == 'x' for c in key_lower):
        return True
    return False

# ─── Main AI Engine ──────────────────────────────────────────

class AIEngine:
    """AI intelligence engine with credit checks, failover, and caching."""

    def __init__(
        self,
        provider_order: Optional[List[str]] = None,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
    ):
        self.token_manager = TokenManager()
        self.provider_order = provider_order or [
            "mistral",
            "deepseek",
            "openrouter",
            "openai",
            "anthropic",
            "google",
            "cohere",
            "huggingface",
            "together",
            "groq",
            "ollama",
        ]
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, dict]] = {}
        self._key_cache: Dict[str, bool] = {}  # provider -> available (cached)

        self.available_providers: List[AIProvider] = []
        self._init_available_providers()

        for provider, config in PROVIDER_REGISTRY.items():
            self.token_manager.set_limit(provider, config.token_limit)

        logger.info(f"AI Engine initialized. Available providers: {[p.value for p in self.available_providers]}")

    def _init_available_providers(self) -> None:
        """Scan environment variables for keys, skipping placeholder ones."""
        for provider, config in PROVIDER_REGISTRY.items():
            if provider == AIProvider.OLLAMA:
                # Ollama doesn't require a key, but we need the host variable
                if os.getenv(config.api_key_env_var):
                    self.available_providers.append(provider)
            else:
                key = os.getenv(config.api_key_env_var)
                if not _is_placeholder_key(key):
                    self.available_providers.append(provider)
                else:
                    logger.debug(f"Skipping {provider.value}: placeholder key detected")

    def get_best_provider(self) -> Optional[AIProvider]:
        """
        Return the first provider in the priority list that is available,
        has credits, and is not exhausted.
        """
        for provider_name in self.provider_order:
            try:
                provider = AIProvider(provider_name)
                if provider in self.available_providers:
                    # Check if we've cached availability
                    if provider_name in self._key_cache and not self._key_cache[provider_name]:
                        continue
                    if not self.token_manager.is_exhausted(provider):
                        # If not cached, assume available and test later
                        return provider
            except ValueError:
                continue
        return None

    async def _check_provider_credits(self, provider: AIProvider) -> bool:
        """Test provider with a lightweight call and cache result."""
        provider_name = provider.value
        if provider_name in self._key_cache:
            return self._key_cache[provider_name]

        config = PROVIDER_REGISTRY[provider]
        try:
            result = await self._call_provider_with_retry(provider, "Hello")
            if result:
                self._key_cache[provider_name] = True
                return True
        except Exception as e:
            error_str = str(e)
            # Permanent failures: invalid key, no credits, wrong endpoint
            if any(code in error_str for code in ["401", "402", "403", "404"]):
                self._key_cache[provider_name] = False
            else:
                # Transient failures (network, timeout) – mark as unavailable but allow retry later
                self._key_cache[provider_name] = False
            return False
        return False

    # ─── Core Analysis ──────────────────────────────────────

    async def analyze_response(
        self,
        url: str,
        status: int,
        content: str,
        headers: Dict[str, str],
        max_content_len: int = 8000,
    ) -> Dict[str, Any]:
        """Analyze HTTP response using the best available LLM."""
        # Cache check
        if self.cache_enabled:
            cache_key = self._compute_cache_key(content, headers)
            if cache_key in self._cache:
                timestamp, cached_result = self._cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    logger.debug("Returning cached analysis for %s", url)
                    return cached_result
                else:
                    del self._cache[cache_key]

        # Find a working provider
        provider = self.get_best_provider()
        if not provider:
            # Try to re‑evaluate all providers
            for p in self.available_providers:
                if await self._check_provider_credits(p):
                    provider = p
                    break
            if not provider:
                logger.warning("No AI provider available for analysis.")
                return {"error": "No AI provider available", "contains_secrets": False}

        logger.info(f"Using AI provider: {provider.value} for {url}")

        # Build prompt
        prompt = self._build_analysis_prompt(url, status, content[:max_content_len], headers)

        try:
            raw_response = await self._call_provider_with_retry(provider, prompt)
            # Simple token estimation – no network call
            tokens_used = self._estimate_tokens(prompt + raw_response)
            self.token_manager.record_usage(provider, tokens_used)
            result = self._parse_analysis(raw_response)
            result["_provider"] = provider.value
            result["_tokens_used"] = tokens_used

            if self.cache_enabled:
                self._cache[cache_key] = (time.time(), result)
            return result

        except Exception as e:
            logger.error(f"Analysis failed for {url}: {e}")
            # Mark provider as failed for future calls
            self._key_cache[provider.value] = False
            return {"error": str(e), "contains_secrets": False, "severity": "Unknown"}

    # ─── Prompt Engineering ──────────────────────────────────

    @staticmethod
    def _build_analysis_prompt(url: str, status: int, content: str, headers: dict) -> str:
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
- "contains_secrets": boolean
- "severity": string: "Critical", "High", "Medium", "Low", or "None"
- "findings": list of strings
- "summary": string, one‑sentence overview
- "recommendation": string, actionable advice

Example:
{{
  "contains_secrets": true,
  "severity": "High",
  "findings": ["Found AWS secret key: AKIA...", "Discovered database password in configuration block"],
  "summary": "AWS credentials and database password exposed.",
  "recommendation": "Remove sensitive values, use environment variables, rotate keys immediately."
}}

Do NOT include any extra text, code fences, or commentary. Only output the JSON.
"""

    # ─── Provider‑specific Calls ────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, ConnectionError)),
        reraise=True,
    )
    async def _call_provider_with_retry(self, provider: AIProvider, prompt: str) -> str:
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

    # ─── Helper for OpenAI-compatible APIs ──────────────────
    @staticmethod
    async def _post_json(url: str, headers: dict, payload: dict, timeout: int) -> Dict:
        """Generic POST helper with JSON response check."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                try:
                    return await resp.json()
                except Exception:
                    text = await resp.text()
                    raise RuntimeError(f"Invalid JSON (status {resp.status}): {text[:200]}")

    # ─── OpenAI (and DeepSeek) ──────────────────────────────
    @staticmethod
    async def _call_openai(config: AIProviderConfig, prompt: str) -> str:
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError(f"Missing API key for {config.name.value}")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": config.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1500,
        }
        url = config.base_url.rstrip('/') + config.chat_endpoint
        data = await AIEngine._post_json(url, headers, payload, config.timeout)
        return data["choices"][0]["message"]["content"]

    # ─── OpenRouter ──────────────────────────────────────────
    @staticmethod
    async def _call_openrouter(config: AIProviderConfig, prompt: str) -> str:
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError("Missing OpenRouter API key")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": config.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1500,
        }
        url = config.base_url.rstrip('/') + config.chat_endpoint
        data = await AIEngine._post_json(url, headers, payload, config.timeout)
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
        url = config.base_url.rstrip('/') + config.chat_endpoint
        data = await AIEngine._post_json(url, headers, payload, config.timeout)
        return data["content"][0]["text"]

    # ─── Google Gemini ──────────────────────────────────────
    @staticmethod
    async def _call_google(config: AIProviderConfig, prompt: str) -> str:
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError("Missing Google API key")
        url = f"{config.base_url.rstrip('/')}/models/{config.default_model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1500},
        }
        data = await AIEngine._post_json(url, {"Content-Type": "application/json"}, payload, config.timeout)
        candidates = data.get("candidates", [])
        if candidates:
            return candidates[0]["content"]["parts"][0]["text"]
        raise RuntimeError("No response from Gemini")

    # ─── Mistral ─────────────────────────────────────────────
    @staticmethod
    async def _call_mistral(config: AIProviderConfig, prompt: str) -> str:
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError("Missing Mistral API key")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": config.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1500,
        }
        url = config.base_url.rstrip('/') + config.chat_endpoint
        data = await AIEngine._post_json(url, headers, payload, config.timeout)
        return data["choices"][0]["message"]["content"]

    # ─── DeepSeek (OpenAI‑compatible) ──────────────────────
    async def _call_deepseek(self, config: AIProviderConfig, prompt: str) -> str:
        return await self._call_openai(config, prompt)

    # ─── Cohere ──────────────────────────────────────────────
    @staticmethod
    async def _call_cohere(config: AIProviderConfig, prompt: str) -> str:
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError("Missing Cohere API key")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": config.default_model,
            "message": prompt,
            "temperature": 0.2,
            "max_tokens": 1500,
        }
        url = config.base_url.rstrip('/') + config.chat_endpoint
        data = await AIEngine._post_json(url, headers, payload, config.timeout)
        return data.get("text", "")

    # ─── Hugging Face ────────────────────────────────────────
    @staticmethod
    async def _call_huggingface(config: AIProviderConfig, prompt: str) -> str:
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError("Missing Hugging Face API key")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1500, "temperature": 0.2, "return_full_text": False},
        }
        url = config.base_url.rstrip('/') + '/' + config.default_model
        data = await AIEngine._post_json(url, headers, payload, config.timeout)
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "")
        elif isinstance(data, dict):
            return data.get("generated_text", "")
        raise RuntimeError("Unexpected response from Hugging Face")

    # ─── Together AI ─────────────────────────────────────────
    @staticmethod
    async def _call_together(config: AIProviderConfig, prompt: str) -> str:
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError("Missing Together AI API key")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": config.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1500,
        }
        url = config.base_url.rstrip('/') + config.chat_endpoint
        data = await AIEngine._post_json(url, headers, payload, config.timeout)
        return data["choices"][0]["message"]["content"]

    # ─── Groq ─────────────────────────────────────────────────
    @staticmethod
    async def _call_groq(config: AIProviderConfig, prompt: str) -> str:
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError("Missing Groq API key")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": config.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1500,
        }
        url = config.base_url.rstrip('/') + config.chat_endpoint
        data = await AIEngine._post_json(url, headers, payload, config.timeout)
        return data["choices"][0]["message"]["content"]

    # ─── Ollama (local) ──────────────────────────────────────
    @staticmethod
    async def _call_ollama(config: AIProviderConfig, prompt: str) -> str:
        host = os.getenv(config.api_key_env_var, "http://localhost:11434")
        url = host.rstrip('/') + config.chat_endpoint
        payload = {
            "model": config.default_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        data = await AIEngine._post_json(url, {"Content-Type": "application/json"}, payload, config.timeout)
        return data.get("response", "")

    # ─── Token Estimation (simple heuristic) ────────────────
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation: ~4 characters per token."""
        return len(text) // 4

    # ─── Response Parsing ────────────────────────────────────
    @staticmethod
    def _parse_analysis(raw: str) -> Dict[str, Any]:
        # Try to extract JSON block
        json_pattern = re.compile(r'\{[^{}]*\}', re.DOTALL)
        matches = json_pattern.findall(raw)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        start = raw.find('{')
        end = raw.rfind('}') + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass

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
        import hashlib
        data = content.encode('utf-8') + str(headers.get('content-type', '')).encode()
        return hashlib.sha256(data).hexdigest()

    # ─── Key Testing ──────────────────────────────────────────
    async def test_all_keys(self) -> Dict[str, Dict[str, Any]]:
        """Test all configured API keys and return detailed status."""
        results = {}
        for provider, config in PROVIDER_REGISTRY.items():
            if provider == AIProvider.OLLAMA:
                key_present = bool(os.getenv(config.api_key_env_var))
            else:
                key = os.getenv(config.api_key_env_var)
                if _is_placeholder_key(key):
                    results[provider.value] = {
                        "available": False,
                        "credits": "🚫 Placeholder key",
                        "error": "Skipped (placeholder detected)"
                    }
                    continue
                key_present = True

            if not key_present:
                results[provider.value] = {
                    "available": False,
                    "credits": "🚫 No key",
                    "error": "Missing API key"
                }
                continue

            try:
                result = await self._call_provider_with_retry(provider, "Hello")
                results[provider.value] = {
                    "available": True,
                    "credits": "✅ Available",
                    "error": None,
                    "response_preview": result[:50] if result else None
                }
            except Exception as e:
                error_str = str(e)
                if "402" in error_str or "insufficient credits" in error_str.lower():
                    results[provider.value] = {
                        "available": False,
                        "credits": "⚠️ No credits",
                        "error": "Insufficient credits"
                    }
                elif "401" in error_str or "403" in error_str:
                    results[provider.value] = {
                        "available": False,
                        "credits": "❌ Invalid key",
                        "error": str(e)[:100]
                    }
                else:
                    results[provider.value] = {
                        "available": False,
                        "credits": "❓ Unknown error",
                        "error": str(e)[:100]
                    }
        return results

# ─── Factory Function ──────────────────────────────────────
def create_ai_engine(**kwargs) -> AIEngine:
    return AIEngine(**kwargs)

# ─── Self‑Test ──────────────────────────────────────────────
if __name__ == "__main__":
    async def main():
        engine = AIEngine()
        print("Available providers:", [p.value for p in engine.available_providers])
        key_status = await engine.test_all_keys()
        print("Key status:")
        for provider, status in key_status.items():
            print(f"  {provider}: {status['credits']} - {status.get('error', 'OK')}")

        sample_url = "https://example.com/.env"
        sample_content = "DB_PASSWORD=supersecret\nAWS_ACCESS_KEY=AKIA123456\n"
        sample_headers = {"content-type": "text/plain"}

        result = await engine.analyze_response(sample_url, 200, sample_content, sample_headers)
        print("\nAnalysis result:", json.dumps(result, indent=2))

    asyncio.run(main())
