"""Voice provider: OpenAI TTS (text-to-speech) + Whisper STT (speech-to-text)."""

import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

# ── defaults ──
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_TTS_VOICE = "coral"
DEFAULT_TTS_FORMAT = "opus"  # Feishu requires opus for voice messages
DEFAULT_STT_MODEL = "gpt-4o-mini-transcribe"

# Voice replies don't work well for very long text or code-heavy output.
TTS_MAX_TEXT_LENGTH = 2000

# ── All available voices (gpt-4o-mini-tts supports all 13) ──
OPENAI_TTS_VOICES: dict[str, str] = {
    "alloy": "Neutral, balanced, versatile (中性平衡)",
    "ash": "Warm, conversational, male (温暖男声)",
    "ballad": "Expressive, melodic, storytelling (富有表现力)",
    "coral": "Warm, engaging, clear female (温暖女声，推荐)",
    "echo": "Smooth, deep male (沉稳男声)",
    "fable": "Expressive, British accent, storyteller (英式口音)",
    "nova": "Warm female, upbeat and friendly (活泼女声)",
    "onyx": "Deep, authoritative male (深沉权威男声)",
    "sage": "Calm, composed, neutral (沉着冷静)",
    "shimmer": "Bright, optimistic female (明亮乐观女声)",
    "verse": "Versatile, expressive (多才多艺)",
    "marin": "Natural, high-quality female (高品质自然女声，推荐)",
    "cedar": "Natural, high-quality male (高品质自然男声，推荐)",
}

# ── Voice directive parsing ──
# Agent can emit [[voice:coral]] or [[voice:coral instructions=开心地说]] in replies.
_VOICE_DIRECTIVE_RE = re.compile(
    r"\[\[voice:(\w+)(?:\s+instructions?=(.+?))?\]\]",
    re.IGNORECASE,
)

# Simple markdown stripping for TTS text preparation.
_HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_ITALIC2_RE = re.compile(r"_(.+?)_")
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_BLOCKQUOTE_RE = re.compile(r"^>\s?", re.MULTILINE)
_HR_RE = re.compile(r"^-{3,}$", re.MULTILINE)
_LIST_MARKER_RE = re.compile(r"^(\s*)[-*+]\s", re.MULTILINE)
_ORDERED_LIST_RE = re.compile(r"^(\s*)\d+\.\s", re.MULTILINE)

# Code block detection: if response is code-heavy, skip TTS.
_CODE_BLOCK_DETECT_RE = re.compile(r"```[\s\S]{40,}?```")


def strip_markdown_for_tts(text: str) -> str:
    """Strip markdown formatting so TTS doesn't read symbols aloud."""
    t = text
    # Remove code blocks entirely (not useful in speech).
    t = _CODE_BLOCK_RE.sub("", t)
    t = _HEADER_RE.sub("", t)
    t = _BOLD_RE.sub(r"\1", t)
    t = _ITALIC_RE.sub(r"\1", t)
    t = _ITALIC2_RE.sub(r"\1", t)
    t = _STRIKE_RE.sub(r"\1", t)
    t = _INLINE_CODE_RE.sub(r"\1", t)
    t = _LINK_RE.sub(r"\1", t)
    t = _BLOCKQUOTE_RE.sub("", t)
    t = _HR_RE.sub("", t)
    t = _LIST_MARKER_RE.sub(r"\1", t)
    t = _ORDERED_LIST_RE.sub(r"\1", t)
    # Collapse excessive whitespace.
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def is_code_heavy(text: str) -> bool:
    """Check if text contains significant code blocks (bad for TTS)."""
    blocks = _CODE_BLOCK_DETECT_RE.findall(text)
    code_chars = sum(len(b) for b in blocks)
    return code_chars > len(text) * 0.4


class VoiceDirective:
    """Parsed [[voice:...]] directive from agent reply."""

    __slots__ = ("voice", "instructions", "mode_switch")

    def __init__(
        self,
        voice: str = "",
        instructions: str = "",
        mode_switch: str = "",  # "on" | "off" | ""
    ):
        self.voice = voice
        self.instructions = instructions
        self.mode_switch = mode_switch  # "on", "off", or "" (no switch)


# Also match [[voice:on]] and [[voice:off]] for mode switching.
_VOICE_MODE_RE = re.compile(r"\[\[voice:(on|off)\]\]", re.IGNORECASE)


def parse_voice_directive(text: str) -> tuple[str, VoiceDirective | None]:
    """Extract [[voice:...]] directives from text.

    Handles:
    - [[voice:on]]  → mode switch to voice
    - [[voice:off]] → mode switch to text
    - [[voice:coral]] → voice selection (implies on)
    - [[voice:coral instructions=...]] → voice + style (implies on)

    Returns (cleaned_text, directive_or_None).
    """
    # Check for mode switch first: [[voice:on]] / [[voice:off]]
    mode_match = _VOICE_MODE_RE.search(text)
    if mode_match:
        mode = mode_match.group(1).lower()
        cleaned = text.replace(mode_match.group(0), "").strip()
        # After stripping mode tag, also check for voice selection tag.
        voice_match = _VOICE_DIRECTIVE_RE.search(cleaned)
        if voice_match:
            voice = voice_match.group(1).lower()
            instructions = (voice_match.group(2) or "").strip()
            if voice in OPENAI_TTS_VOICES:
                cleaned = cleaned.replace(voice_match.group(0), "").strip()
                return cleaned, VoiceDirective(voice=voice, instructions=instructions, mode_switch=mode)
        return cleaned, VoiceDirective(mode_switch=mode)

    # Check for voice selection: [[voice:coral]] / [[voice:coral instructions=...]]
    m = _VOICE_DIRECTIVE_RE.search(text)
    if not m:
        return text, None
    voice = m.group(1).lower()
    instructions = (m.group(2) or "").strip()
    if voice not in OPENAI_TTS_VOICES:
        logger.warning(f"Unknown voice '{voice}', ignoring directive")
        return text.replace(m.group(0), "").strip(), None
    cleaned = text.replace(m.group(0), "").strip()
    # Selecting a voice implies voice mode on.
    return cleaned, VoiceDirective(voice=voice, instructions=instructions, mode_switch="on")


def build_voice_options_hint() -> str:
    """Build a system prompt section listing available voices."""
    lines = ["Available voices (use [[voice:name]] to choose, or [[voice:name instructions=语气描述]]):"]
    for name, desc in OPENAI_TTS_VOICES.items():
        lines.append(f"  - {name}: {desc}")
    lines.append("")
    lines.append("You can also provide speech instructions to control tone, emotion, accent, speed, etc.")
    lines.append("Example: [[voice:coral instructions=用温柔开心的语气说]]")
    lines.append("Example: [[voice:onyx instructions=Speak slowly with a deep authoritative tone]]")
    return "\n".join(lines)


class OpenAIVoiceProvider:
    """OpenAI-based TTS and STT provider.

    Uses:
    - /v1/audio/speech  for text-to-speech (gpt-4o-mini-tts)
    - /v1/audio/transcriptions  for speech-to-text (Whisper)
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        tts_model: str = DEFAULT_TTS_MODEL,
        tts_voice: str = DEFAULT_TTS_VOICE,
        stt_model: str = DEFAULT_STT_MODEL,
    ):
        self.api_key = (
            api_key
            or os.environ.get("GETALL_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self.api_base = (
            api_base
            or os.environ.get("GETALL_OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.stt_model = stt_model

        # Media directory for voice files.
        self.voice_dir = Path.home() / ".getall" / "media" / "voice"
        self.voice_dir.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.api_key)

    def _make_output_path(self, prefix: str = "tts", ext: str = ".opus") -> Path:
        """Generate a timestamped output path."""
        ts = int(time.time() * 1000)
        return self.voice_dir / f"{prefix}_{ts}{ext}"

    async def tts(
        self,
        text: str,
        output_path: Path | None = None,
        *,
        voice: str | None = None,
        model: str | None = None,
        instructions: str | None = None,
        response_format: str = DEFAULT_TTS_FORMAT,
        speed: float = 1.0,
    ) -> Path:
        """Convert text to speech audio file.

        Args:
            text: Text to synthesize.
            output_path: Where to save (auto-generated if None).
            voice: TTS voice (default: coral).
            model: TTS model (default: gpt-4o-mini-tts).
            instructions: Speech style instructions (gpt-4o-mini-tts only).
            response_format: Audio format (opus, mp3, etc.).
            speed: Playback speed 0.25–4.0.

        Returns:
            Path to the generated audio file.

        Raises:
            RuntimeError: If API call fails.
        """
        if not self.api_key:
            raise RuntimeError("OpenAI API key not configured for TTS")

        if not output_path:
            ext = f".{response_format}" if response_format != "pcm" else ".pcm"
            output_path = self._make_output_path("tts", ext)

        url = f"{self.api_base}/audio/speech"

        body: dict[str, Any] = {
            "model": model or self.tts_model,
            "input": text,
            "voice": voice or self.tts_voice,
            "response_format": response_format,
            "speed": speed,
        }
        # instructions only supported by gpt-4o-mini-tts
        if instructions and (body["model"].startswith("gpt-4o")):
            body["instructions"] = instructions

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=60.0,
            )
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(response.content)
            logger.debug(
                f"TTS: {len(text)} chars → {output_path.name} "
                f"(voice={body['voice']}, {len(response.content)} bytes)"
            )
            return output_path

    async def stt(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
    ) -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (opus, ogg, mp3, wav, etc.).
            language: Optional ISO-639-1 language hint (e.g. "zh", "en").

        Returns:
            Transcribed text, or empty string on failure.
        """
        if not self.api_key:
            logger.warning("OpenAI API key not configured for STT")
            return ""

        path = Path(audio_path)
        if not path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return ""

        url = f"{self.api_base}/audio/transcriptions"

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {"file": (path.name, f)}
                    data: dict[str, str] = {"model": self.stt_model}
                    if language:
                        data["language"] = language

                    response = await client.post(
                        url,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        files=files,
                        data=data,
                        timeout=60.0,
                    )
                    response.raise_for_status()
                    result = response.json()
                    text = result.get("text", "")
                    logger.debug(f"STT: {path.name} → {len(text)} chars")
                    return text

        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            return ""
