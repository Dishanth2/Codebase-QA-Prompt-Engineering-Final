"""Minimal multi-provider chat client (no vendor SDKs needed).

Providers
---------
* OpenAI-compatible HTTP API: gemini, groq, openai, openrouter, ollama
* Anthropic Messages API: anthropic
* offline: a deterministic stand-in used when no API key is configured, so the
  UI, the agent loop and the tests still run end-to-end.

Responses are cached on disk (data/cache/llm) keyed by the full request, so
re-running an evaluation or a demo question costs nothing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from typing import Callable

import httpx

from . import config

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


# Fast, reliable timeouts: 10s connect, 45s read
_TIMEOUT = httpx.Timeout(connect=10.0, read=45.0, write=10.0, pool=10.0)

# HTTP codes that are worth retrying for transient server issues
_RETRYABLE = {429, 500, 502, 503, 504, 529}


def _clean_error_message(text: str) -> str:
    """Extract a safe, human-readable error message from JSON error response."""
    try:
        data = json.loads(text)
        if isinstance(data, list) and data:
            data = data[0]
        if isinstance(data, dict) and "error" in data:
            err = data["error"]
            if isinstance(err, dict) and "message" in err:
                return err["message"]
            if isinstance(err, str):
                return err
    except Exception:
        pass
    return text.strip()[:300]


def _mask_key(text: str, key: str | None = None) -> str:
    """Safely mask API key in any error or log message."""
    if not key or len(key) < 8:
        return text
    return text.replace(key, key[:4] + "..." + key[-4:])


class LLM:
    def __init__(self, settings: config.LLMSettings | None = None,
                 offline_policy: Callable[[list[Message]], str] | None = None):
        self.s = settings or config.llm_settings()
        self.offline_policy = offline_policy
        self.calls = 0
        self.cache_hits = 0
        if self.s.provider not in ("offline", "ollama") and not self.s.api_key:
            raise LLMError(f"No API key for provider {self.s.provider!r}. "
                           f"Set {config.LLM_PRESETS[self.s.provider][2]} in .env")
        log.info("LLM initialized: provider=%s, model=%s, base_url=%s, key_set=%s",
                 self.s.provider, self.s.model, self.s.base_url, bool(self.s.api_key))

    @property
    def name(self) -> str:
        return f"{self.s.provider}:{self.s.model}"

    @property
    def is_offline(self) -> bool:
        return self.s.provider == "offline"

    # ------------------------------------------------------------------ API
    def chat(self, messages: list[Message], stop: list[str] | None = None,
             max_tokens: int | None = None, temperature: float | None = None,
             json_mode: bool = False) -> str:
        if self.is_offline:
            if not self.offline_policy:
                raise LLMError("Offline mode has no policy for this task; configure an LLM API key.")
            return self.offline_policy(messages)

        payload = {
            "messages": messages, "stop": stop or [], "model": self.s.model,
            "max_tokens": max_tokens or self.s.max_tokens,
            "temperature": self.s.temperature if temperature is None else temperature,
            "json": json_mode, "provider": self.s.provider,
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cache_f = config.CACHE_DIR / "llm" / f"{key}.json"
        if config.LLM_CACHE and cache_f.exists():
            self.cache_hits += 1
            return json.loads(cache_f.read_text())["text"]

        text = self._with_retries(lambda: self._call(payload))
        if stop:  # some providers ignore stop sequences: enforce them client-side
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        self.calls += 1
        if config.LLM_CACHE:
            cache_f.parent.mkdir(parents=True, exist_ok=True)
            cache_f.write_text(json.dumps({"text": text, "model": self.name}))
        return text

    # ------------------------------------------------------------ internals
    def _with_retries(self, fn, attempts: int = 3) -> str:
        delay = 2.0
        for i in range(attempts):
            try:
                return fn()
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                raw_body = e.response.text
                err_msg = _mask_key(_clean_error_message(raw_body), self.s.api_key)
                log.warning("LLM %s HTTP %d (attempt %d/%d): %s",
                            self.name, code, i + 1, attempts, err_msg)

                # Check for hard quota limit exhaustion (e.g. daily free-tier limit)
                retry_after_str = e.response.headers.get("retry-after")
                retry_after = float(retry_after_str) if retry_after_str and retry_after_str.isdigit() else None
                is_quota_exhausted = (
                    code == 429 and (
                        (retry_after and retry_after > 30) or
                        "RESOURCE_EXHAUSTED" in raw_body or
                        "limit exceeded" in raw_body.lower() or
                        "quota" in err_msg.lower()
                    )
                )
                if is_quota_exhausted:
                    # Do not hang waiting on daily quota exhaustion; fail fast with clear message
                    raise LLMError(f"{self.name} rate limited (HTTP 429): {err_msg}") from e

                if code in _RETRYABLE and i < attempts - 1:
                    wait = min(retry_after or delay, 10.0)
                    time.sleep(wait + random.random())
                    delay *= 2
                    continue

                if code == 400:
                    raise LLMError(f"{self.name} bad request (HTTP 400): {err_msg}") from e
                if code == 401:
                    raise LLMError(f"{self.name} authentication failed (HTTP 401): check your API key") from e
                if code == 403:
                    raise LLMError(f"{self.name} access denied (HTTP 403): {err_msg}") from e
                if code == 404:
                    raise LLMError(f"{self.name} model not found (HTTP 404): {err_msg}") from e
                if code == 429:
                    raise LLMError(f"{self.name} rate limited (HTTP 429): {err_msg}") from e
                raise LLMError(f"{self.name} HTTP {code}: {err_msg}") from e
            except (httpx.TransportError, httpx.TimeoutException) as e:
                log.warning("LLM %s connection error (attempt %d/%d): %s",
                            self.name, i + 1, attempts, e)
                if i < attempts - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError(f"{self.name} connection failed after {attempts} attempts: {e}") from e
        raise LLMError("unreachable")

    def _call(self, p: dict) -> str:
        if self.s.provider == "anthropic":
            return self._anthropic(p)
        return self._openai_compatible(p)

    def _openai_compatible(self, p: dict) -> str:
        body = {"model": p["model"], "messages": p["messages"], "temperature": p["temperature"],
                "max_tokens": p["max_tokens"]}
        if p["stop"]:
            body["stop"] = p["stop"][:4]
        if p["json"]:
            body["response_format"] = {"type": "json_object"}
        if self.s.reasoning_effort:
            body["reasoning_effort"] = self.s.reasoning_effort
        headers = {"Content-Type": "application/json"}
        if self.s.api_key:
            headers["Authorization"] = f"Bearer {self.s.api_key}"
        url = f"{self.s.base_url.rstrip('/')}/chat/completions"
        r = httpx.post(url, json=body, headers=headers, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        try:
            choices = data.get("choices") or []
            if not choices:
                raise LLMError(f"Empty choices in response: {str(data)[:200]}")
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if content is not None:
                return content
            if "tool_calls" in msg and msg["tool_calls"]:
                tc = msg["tool_calls"][0]
                fn = tc.get("function", {})
                return f"Action: {fn.get('name')}\nAction Input: {fn.get('arguments', '{}')}"
            return ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected response: {str(data)[:200]}") from e

    def _anthropic(self, p: dict) -> str:
        system = "\n\n".join(m["content"] for m in p["messages"] if m["role"] == "system")
        msgs = [m for m in p["messages"] if m["role"] != "system"]
        body = {"model": p["model"], "max_tokens": p["max_tokens"], "messages": msgs,
                "temperature": p["temperature"]}
        if system:
            body["system"] = system
        if p["stop"]:
            body["stop_sequences"] = p["stop"]
        r = httpx.post(f"{self.s.base_url.rstrip('/')}/messages", json=body, timeout=_TIMEOUT, headers={
            "x-api-key": self.s.api_key, "anthropic-version": "2023-06-01",
            "content-type": "application/json"})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []))


def make_llm(provider: str | None = None, model: str | None = None,
             offline_policy=None) -> LLM:
    return LLM(config.llm_settings(provider, model), offline_policy=offline_policy)