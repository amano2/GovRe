"""OpenRouter API LLM Client Wrapper with retry, multi-model fallback chain,
explicit response source tracking, and transparent failure handling.

IMPORTANT: This client does NOT include any local heuristic fallback.
If all API models fail, it returns a clearly-marked API_FAILURE response
so that the evaluation pipeline can exclude or separately report these cases.
This prevents contamination of System B (retrieval-grounded) results with
disguised rule-based logic.
"""

import os
import sys
import json
import time
import hashlib
import urllib.request
import urllib.error
import ssl
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field, asdict

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
    FALLBACK_OPENROUTER_MODELS,
)


@dataclass
class LLMResponse:
    """Structured response from LLM client with full provenance metadata."""
    content: str
    response_source: str  # "llm" or "api_failure"
    model_used: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    attempts: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMClient:
    """OpenRouter LLM client with multi-model fallback chain and explicit
    response source tracking for research integrity.

    Key design decision: NO local heuristic fallback. If the API is unreachable
    or all models fail, the client returns a structured API_FAILURE response
    with response_source='api_failure'. The evaluation pipeline must handle
    these explicitly (exclude, retry, or report separately).
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "")
        self.primary_model = model or DEFAULT_OPENROUTER_MODEL
        # Build ordered fallback chain, deduplicating
        seen = set()
        self.models_to_try: List[str] = []
        for m in [self.primary_model] + FALLBACK_OPENROUTER_MODELS:
            if m not in seen:
                self.models_to_try.append(m)
                seen.add(m)
        self.base_url = OPENROUTER_BASE_URL
        
        # Disk cache path for transparent, deterministic research caching
        self.cache_path = PROJECT_ROOT / "results" / "llm_cache.json"
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception:
            pass

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are an expert enterprise governance AI that evaluates operational requests against policy documents.",
        temperature: float = 0.0,
        max_retries: int = 2,
        seed: Optional[int] = None,
        timeout_seconds: float = 30.0,
    ) -> str:
        """Generate text completion via OpenRouter. Returns raw content string.

        For research evaluation, prefer generate_with_metadata() which returns
        full provenance information including response_source.
        """
        response = self.generate_with_metadata(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_retries=max_retries,
            seed=seed,
            timeout_seconds=timeout_seconds,
        )
        return response.content

    def generate_with_metadata(
        self,
        prompt: str,
        system_prompt: str = "You are an expert enterprise governance AI that evaluates operational requests against policy documents.",
        temperature: float = 0.0,
        max_retries: int = 2,
        seed: Optional[int] = None,
        timeout_seconds: float = 30.0,
    ) -> LLMResponse:
        """Generate text completion with full provenance metadata.

        Returns LLMResponse with response_source indicating whether the
        response came from the LLM API ('llm') or failed ('api_failure').
        """
        errors: List[str] = []
        total_attempts = 0

        # Check transparent deterministic research cache
        cache_key = hashlib.sha256(
            f"{prompt}__{system_prompt}__{temperature}__{seed}".encode("utf-8")
        ).hexdigest()
        if cache_key in self._cache and self._cache[cache_key].get("response_source") == "llm":
            cached = self._cache[cache_key]
            return LLMResponse(
                content=cached["content"],
                response_source="llm",
                model_used=cached.get("model_used", self.primary_model),
                finish_reason=cached.get("finish_reason", "stop"),
                prompt_tokens=cached.get("prompt_tokens", 0),
                completion_tokens=cached.get("completion_tokens", 0),
                total_tokens=cached.get("total_tokens", 0),
                latency_ms=cached.get("latency_ms", 1.0),
                attempts=1,
                errors=[],
            )

        if not self.api_key:
            return LLMResponse(
                content=json.dumps({
                    "decision": "FLAG",
                    "citation_chunk_id": "api_failure",
                    "citation_text": "",
                    "reasoning": "No API key configured. Cannot make LLM governance judgment."
                }),
                response_source="api_failure",
                attempts=0,
                errors=["OPENROUTER_API_KEY not set or empty"],
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/governance-grounding-research",
            "X-Title": "GovRe-Governance-Research",
            "User-Agent": "GovRe-Research-Agent/1.0",
        }

        ssl_ctx = ssl.create_default_context()

        for model_name in self.models_to_try:
            for attempt in range(max_retries):
                total_attempts += 1
                start_time = time.perf_counter()
                try:
                    payload: Dict[str, Any] = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": temperature,
                    }
                    if seed is not None:
                        payload["seed"] = seed

                    payload_bytes = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(
                        f"{self.base_url}/chat/completions",
                        data=payload_bytes,
                        headers=headers,
                        method="POST",
                    )

                    with urllib.request.urlopen(req, timeout=timeout_seconds, context=ssl_ctx) as response:
                        raw_body = response.read().decode("utf-8")
                        latency_ms = (time.perf_counter() - start_time) * 1000
                        data = json.loads(raw_body)

                        if "error" in data:
                            error_msg = data["error"].get("message", str(data["error"]))
                            errors.append(f"[{model_name}] API error in response: {error_msg}")
                            break  # Try next model

                        choices = data.get("choices", [])
                        if not choices:
                            errors.append(f"[{model_name}] No choices returned in response: {raw_body[:200]}")
                            break

                        content = choices[0].get("message", {}).get("content", "").strip()
                        usage = data.get("usage", {})
                        finish_reason = choices[0].get("finish_reason", "stop")

                        res_obj = LLMResponse(
                            content=content,
                            response_source="llm",
                            model_used=model_name,
                            finish_reason=finish_reason,
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                            total_tokens=usage.get("total_tokens", 0),
                            latency_ms=round(latency_ms, 2),
                            attempts=total_attempts,
                            errors=errors,
                        )
                        # Save successful LLM completion to transparent cache
                        self._cache[cache_key] = res_obj.to_dict()
                        self._save_cache()
                        return res_obj

                except urllib.error.HTTPError as http_err:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    status_code = http_err.code
                    err_body = ""
                    try:
                        err_body = http_err.read().decode("utf-8")
                    except Exception:
                        pass

                    if status_code == 429:
                        errors.append(f"[{model_name}] Rate limited (429) on attempt {attempt+1}")
                        time.sleep(1.2 * (attempt + 1))  # Backoff
                        continue  # Retry or rotate
                    elif status_code in (401, 403):
                        errors.append(f"[{model_name}] Auth error ({status_code}): {err_body[:200]}")
                        break
                    else:
                        errors.append(f"[{model_name}] HTTP {status_code}: {err_body[:200]}")
                        break  # Try next model

                except (urllib.error.URLError, TimeoutError, TimeoutException if 'TimeoutException' in locals() else Exception) as net_err:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    errors.append(f"[{model_name}] Network/timeout error ({latency_ms:.0f}ms): {str(net_err)[:200]}")
                    time.sleep(0.5)
                    continue

                except Exception as e:
                    errors.append(f"[{model_name}] Unexpected error: {type(e).__name__}: {str(e)[:200]}")
                    break

        # All models exhausted — return explicit failure (NO hidden rule engine)
        return LLMResponse(
            content=json.dumps({
                "decision": "FLAG",
                "citation_chunk_id": "api_failure",
                "citation_text": "",
                "reasoning": f"All API models exhausted after {total_attempts} attempts. Errors: {'; '.join(errors[-3:])}"
            }),
            response_source="api_failure",
            attempts=total_attempts,
            errors=errors,
        )
