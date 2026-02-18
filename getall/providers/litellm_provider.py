"""LiteLLM provider implementation for multi-provider support."""

import json
import os
import re
from typing import Any

import litellm
from litellm import acompletion

from getall.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from getall.providers.registry import find_by_model, find_gateway


# ---------------------------------------------------------------------------
# Manual pricing fallback for models litellm doesn't recognise
# (gateway-proxied, custom IDs, very new models).
#
# Format: regex pattern → (input $/1M tokens, output $/1M tokens)
# Patterns are matched case-insensitively against the *resolved* model name.
# Order matters — first match wins.
# Prices sourced from official pricing pages (2026-02).
# ---------------------------------------------------------------------------

_COST_FALLBACK: list[tuple[re.Pattern[str], float, float]] = [
    # ── Anthropic Claude ──
    # Claude Opus 4.6 ($5/$25) — must be before Opus 4 to match first
    (re.compile(r"opus.?4\.?6|cl_opus_4\.?6", re.I), 5.00, 25.00),
    # Claude 4.5 Sonnet ($3/$15) — any gateway alias
    (re.compile(r"sonnet.?4\.?5|cl_sonnet_4\.?5", re.I), 3.00, 15.00),
    # Claude Opus 4 ($15/$75)
    (re.compile(r"opus.?4|cl_opus_4", re.I), 15.00, 75.00),
    # Claude Sonnet 4 ($3/$15)
    (re.compile(r"sonnet.?4|cl_sonnet_4", re.I), 3.00, 15.00),
    # Claude 3.5 Sonnet ($3/$15)
    (re.compile(r"sonnet.?3\.?5|cl_sonnet_3\.?5", re.I), 3.00, 15.00),

    # ── OpenAI ──
    # GPT-4o ($2.50/$10)
    (re.compile(r"gpt-?4o(?!-mini)", re.I), 2.50, 10.00),
    # GPT-4o-mini ($0.15/$0.60)
    (re.compile(r"gpt-?4o-?mini", re.I), 0.15, 0.60),
    # GPT-5.2 / GPT-Codex-5.2 ($1.75/$14) — must be before GPT-5 generic
    (re.compile(r"gpt-?5\.?2|gpt-?codex-?5\.?2", re.I), 1.75, 14.00),
    # GPT-5 / GPT-Codex-5 ($1.25/$10)
    (re.compile(r"gpt-?5|gpt-?codex-?5", re.I), 1.25, 10.00),

    # ── DeepSeek ──
    # DeepSeek-R1 ($0.55/$2.19)
    (re.compile(r"deepseek-?r1", re.I), 0.55, 2.19),
    # DeepSeek-V3 / deepseek-chat ($0.30/$1.20)
    (re.compile(r"deepseek-?v3|deepseek-?chat", re.I), 0.30, 1.20),

    # ── Google Gemini ──
    # Gemini 2.5 Pro ($1.25/$10)
    (re.compile(r"gemini.?2\.?5.?pro", re.I), 1.25, 10.00),
    # Gemini 2.5 Flash ($0.30/$2.50)
    (re.compile(r"gemini.?2\.?5.?flash", re.I), 0.30, 2.50),

    # ── MiniMax ──
    # MiniMax M2.5-highspeed ($0.60/$2.40) — must be before standard M2.5
    (re.compile(r"minimax.?m2\.?5.?high", re.I), 0.60, 2.40),
    # MiniMax M2.5 standard ($0.30/$1.20)
    (re.compile(r"minimax.?m2\.?5", re.I), 0.30, 1.20),

    # ── Moonshot Kimi ──
    # Kimi K2.5 ($0.60/$3.00)
    # Moonshot native: "kimi-k2.5" (dot); Volcengine Ark: "kimi-k2-5-260127" (dash)
    (re.compile(r"kimi.?k2[.\-]?5", re.I), 0.60, 3.00),

    # ── Alibaba Qwen ──
    # Qwen 3 235B ($0.22/$0.88)
    (re.compile(r"qwen.?3", re.I), 0.22, 0.88),
]


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.
    
    Supports OpenRouter, Anthropic, OpenAI, Gemini, MiniMax, and many other providers through
    a unified interface.  Provider-specific logic is driven by the registry
    (see providers/registry.py) — no if-elif chains needed here.
    """
    
    def __init__(
        self, 
        api_key: str | None = None, 
        api_base: str | None = None,
        default_model: str = "",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        # Per-model credential overrides: model_name → (api_key, api_base | None)
        self._model_credentials: dict[str, tuple[str, str | None]] = {}
        
        # Detect gateway / local deployment.
        # provider_name (from config key) is the primary signal;
        # api_key / api_base are fallback for auto-detection.
        self._gateway = find_gateway(provider_name, api_key, api_base)
        
        # Configure environment variables
        if api_key:
            self._setup_env(api_key, api_base, default_model)
        
        # NOTE: Do NOT set litellm.api_base globally — it would bleed into
        # per-model calls that target different endpoints.  Instead, api_base
        # is passed per-call in kwargs (see chat()).
        
        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True
        # Drop unsupported parameters for providers (e.g., gpt-5 rejects some params)
        litellm.drop_params = True

    def register_model_credentials(
        self, model: str, api_key: str, api_base: str | None = None,
    ) -> None:
        """Register model-specific credentials for multi-provider support.

        When ``chat()`` is called with *model*, these credentials are used
        instead of the default provider's api_key / api_base.
        """
        self._model_credentials[model] = (api_key, api_base)
    
    def _setup_env(self, api_key: str, api_base: str | None, model: str) -> None:
        """Set environment variables based on detected provider."""
        spec = self._gateway or find_by_model(model)
        if not spec:
            return

        # Gateway/local overrides existing env; standard provider doesn't
        if self._gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)

        # Resolve env_extras placeholders:
        #   {api_key}  → user's API key
        #   {api_base} → user's api_base, falling back to spec.default_api_base
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key)
            resolved = resolved.replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)
    
    def _resolve_model(self, model: str, *, skip_gateway: bool = False) -> str:
        """Resolve model name by applying provider/gateway prefixes.

        Args:
            model: Raw model identifier.
            skip_gateway: If True, bypass gateway prefix and resolve via
                the model's own provider spec. Used when per-model credentials
                point to a different endpoint than the default gateway.
        """
        if self._gateway and not skip_gateway:
            # Gateway mode: apply gateway prefix, skip provider-specific prefixes
            prefix = self._gateway.litellm_prefix
            if self._gateway.strip_model_prefix:
                model = model.split("/")[-1]
            if prefix and not model.startswith(f"{prefix}/"):
                model = f"{prefix}/{model}"
            return model
        
        # Standard mode: auto-prefix for known providers
        spec = find_by_model(model)
        if spec and spec.litellm_prefix:
            if not any(model.startswith(s) for s in spec.skip_prefixes):
                model = f"{spec.litellm_prefix}/{model}"
        
        return model
    
    def _apply_model_overrides(self, model: str, kwargs: dict[str, Any]) -> None:
        """Apply model-specific parameter overrides from the registry."""
        model_lower = model.lower()
        spec = find_by_model(model)
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    return
    
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 65536,
        temperature: float = 0.7,
        reasoning_effort: str = "",
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4-5').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (kept for backward compatibility).
            reasoning_effort: Reasoning effort level ("none","low","medium","high","xhigh").
                Empty string means no reasoning param is sent.
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        raw_model = model or self.default_model

        # Resolve credentials BEFORE model-name resolution so the lookup
        # matches the key used at registration time (raw model name).
        creds = self._model_credentials.get(raw_model)

        # Skip gateway prefix when per-model credentials use a DIFFERENT
        # api_key — meaning the model is served by a different provider.
        # Models on the same gateway (same api_key) keep the gateway prefix.
        skip_gw = bool(creds and creds[0] and creds[0] != self.api_key)
        model = self._resolve_model(raw_model, skip_gateway=skip_gw)
        
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        
        # Reasoning effort (OpenAI Responses API / o-series / gpt-5+)
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        
        # Apply model-specific overrides (e.g. kimi-k2.5 temperature)
        self._apply_model_overrides(model, kwargs)
        if creds:
            cred_key, cred_base = creds
            if cred_key:
                kwargs["api_key"] = cred_key
            if cred_base:
                kwargs["api_base"] = cred_base
            # No extra_headers for overridden models — they use their own provider
        else:
            # Pass api_key directly — more reliable than env vars alone
            if self.api_key:
                kwargs["api_key"] = self.api_key
            # Pass api_base for custom endpoints
            if self.api_base:
                kwargs["api_base"] = self.api_base
            # Pass extra headers (e.g. APP-Code for AiHubMix)
            if self.extra_headers:
                kwargs["extra_headers"] = self.extra_headers
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        try:
            response = await acompletion(**kwargs)
            return self._parse_response(response, resolved_model=model)
        except Exception as e:
            from loguru import logger
            logger.error(f"LLM call failed ({model}): {e}")
            # Return a user-friendly error; raw exception details stay in logs only.
            user_msg = self._friendly_error(e)
            return LLMResponse(
                content=user_msg,
                finish_reason="error",
            )

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        """Map raw LLM exceptions to user-friendly messages."""
        raw = str(exc).lower()
        if "max_output_tokens" in raw or "max_tokens" in raw or "length" in raw:
            return "回复内容太长被截断了，请换个更简短的问法再试一次。"
        if "rate_limit" in raw or "429" in raw:
            return "请求太频繁了，稍等几秒再试。"
        if "context_length" in raw or "context window" in raw:
            return "对话太长了，我需要清理一下上下文。请重新提问。"
        if "timeout" in raw:
            return "请求超时了，请稍后再试。"
        if "connection" in raw or "connect" in raw:
            return "网络连接出了点问题，请稍后再试。"
        if "authentication" in raw or "401" in raw or "403" in raw:
            return "认证出了点问题，请联系管理员。"
        return "服务暂时出了点问题，请稍后再试。"
    
    def _parse_response(self, response: Any, *, resolved_model: str = "") -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string if needed
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        # Cost estimation: litellm built-in → manual fallback
        cost_usd = self._estimate_cost(response, resolved_model, usage)
        
        reasoning_content = getattr(message, "reasoning_content", None)
        
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
            cost_usd=cost_usd,
            model=resolved_model or getattr(response, "model", ""),
        )
    
    @staticmethod
    def _estimate_cost(
        response: Any,
        resolved_model: str,
        usage: dict[str, int],
    ) -> float:
        """Estimate USD cost for a completion.

        Strategy:
        1. Try ``litellm.completion_cost()`` — works for standard models.
        2. Fall back to ``_COST_FALLBACK`` regex table for gateway-proxied
           or unrecognised model names.
        3. Return 0 if no pricing info at all (tokens are still recorded).
        """
        # 1. litellm built-in
        try:
            cost = float(litellm.completion_cost(completion_response=response))
            if cost > 0:
                return cost
        except Exception:
            pass

        # 2. Manual fallback
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        if prompt_tokens == 0 and completion_tokens == 0:
            return 0.0

        for pattern, input_per_m, output_per_m in _COST_FALLBACK:
            if pattern.search(resolved_model):
                return (
                    prompt_tokens * input_per_m / 1_000_000
                    + completion_tokens * output_per_m / 1_000_000
                )

        return 0.0

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
