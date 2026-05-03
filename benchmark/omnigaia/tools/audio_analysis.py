"""OmniGAIA benchmark - Audio analysis tool.

The data field in input_audio supports both base64-encoded audio and local file paths.
The API gateway has a request-body size limit; long audio must be split into segments (≤ 30s each).
"""
from __future__ import annotations

import base64
import io
import os
import struct
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from base.agent.base_action import BaseAction
from base.engine.async_llm import LLMsConfig, llm_retry
from base.engine.logs import logger

# Max seconds per audio segment (30s WAV ~ 0.92 MB, base64 ~ 1.22 MB, within gateway limit)
MAX_SEGMENT_SECONDS = 30


def _parse_wav_header(wav_data: bytes) -> dict:
    """Parse WAV file header; return fmt_data, data_start, data_size, byte_rate, block_align."""
    if wav_data[:4] != b'RIFF' or wav_data[8:12] != b'WAVE':
        raise ValueError("Not a valid WAV file")

    pos = 12
    fmt_data = None
    data_start = None
    data_size = None
    byte_rate = None
    block_align = None

    while pos < len(wav_data) - 8:
        chunk_id = wav_data[pos:pos + 4]
        chunk_size = struct.unpack('<I', wav_data[pos + 4:pos + 8])[0]

        if chunk_id == b'fmt ':
            fmt_data = wav_data[pos:pos + 8 + chunk_size]
            _, _, _, byte_rate, block_align, _ = struct.unpack(
                '<HHIIHH', wav_data[pos + 8:pos + 24]
            )
        elif chunk_id == b'data':
            data_start = pos + 8
            data_size = chunk_size
            break

        pos += 8 + chunk_size
        if chunk_size % 2 == 1:  # WAV chunk byte alignment
            pos += 1

    if fmt_data is None or data_start is None or byte_rate is None:
        raise ValueError("Malformed WAV file: unable to parse fmt/data chunk")

    return {
        "fmt_data": fmt_data,
        "data_start": data_start,
        "data_size": data_size,
        "byte_rate": byte_rate,
        "block_align": block_align,
    }


def _build_wav_segment(header: dict, wav_data: bytes, offset: int, length: int) -> bytes:
    """Build a standalone WAV file from a slice of the raw WAV data."""
    fmt_data = header["fmt_data"]
    data_start = header["data_start"]
    block_align = header["block_align"]

    # Ensure block_align alignment
    length = (length // block_align) * block_align
    if length == 0:
        return b""

    output = io.BytesIO()
    total_size = 4 + len(fmt_data) + 8 + length
    output.write(b'RIFF')
    output.write(struct.pack('<I', total_size))
    output.write(b'WAVE')
    output.write(fmt_data)
    output.write(b'data')
    output.write(struct.pack('<I', length))
    output.write(wav_data[data_start + offset:data_start + offset + length])
    return output.getvalue()


def get_wav_duration(wav_data: bytes) -> float:
    """Get WAV file duration in seconds."""
    try:
        header = _parse_wav_header(wav_data)
        return header["data_size"] / header["byte_rate"]
    except (ValueError, ZeroDivisionError):
        return 0.0


def split_wav_segments(wav_data: bytes, segment_seconds: float = 30) -> list[bytes]:
    """Split a WAV file into segments of at most segment_seconds seconds each."""
    header = _parse_wav_header(wav_data)
    byte_rate = header["byte_rate"]
    block_align = header["block_align"]
    data_size = header["data_size"]

    segment_bytes = int(segment_seconds * byte_rate)
    segment_bytes = (segment_bytes // block_align) * block_align

    segments = []
    offset = 0
    while offset < data_size:
        chunk_len = min(segment_bytes, data_size - offset)
        chunk_len = (chunk_len // block_align) * block_align
        if chunk_len == 0:
            break
        seg = _build_wav_segment(header, wav_data, offset, chunk_len)
        if seg:
            segments.append(seg)
        offset += chunk_len

    return segments


class ParseAudioAction(BaseAction):
    """Audio parsing tool for OmniGAIA.

    Uses gpt-4o-audio-preview and input_audio format.
    Long audio files (>30s) are automatically split into segments and summarized.
    """

    name: str = "ParseAudioAction"
    description: str = (
        "Transcribe and analyze an audio clip using the gpt-4o-audio-preview model. "
        "Supports long audio files by automatically splitting into segments. "
        "Uses model_config.yaml for API configuration."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Instruction that describes how to handle or summarize the audio content",
            },
            "audio_path": {
                "type": "string",
                "description": "Local file path that points to the audio resource (e.g. mp3, wav)",
            },
        },
        "required": ["query", "audio_path"],
        "additionalProperties": False,
    }

    # Model for audio transcription (supports input_audio format)
    AUDIO_MODEL: str = "gpt-4o-audio-preview"
    # Model for summarizing multi-segment transcriptions
    SUMMARY_MODEL: str = "gpt-5"

    def _get_llm_config(self, model_name: str) -> tuple:
        """Get LLM configuration from model_config.yaml or environment variables."""
        try:
            llms_config = LLMsConfig.default()
            model_config = llms_config.get(model_name)
            if model_config:
                return (model_config.key, model_config.base_url, model_config.model)
        except Exception as e:
            logger.warning(f"[ParseAudio] Failed to load {model_name} config from model_config.yaml: {e}")

        # Fallback to environment variables
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        return (api_key, base_url, model_name)

    async def _transcribe_segment(
        self, client: AsyncOpenAI, model: str, audio_b64: str, fmt: str, query: str,
    ) -> str:
        """Transcribe a single audio segment using gemini-3.1-pro-preview."""
        completion = await llm_retry(
            client.chat.completions.create,
            model=model,
            #modalities=["text"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": fmt,
                        },
                    },
                ],
            }],
            label=f"ParseAudio._transcribe_segment({model})",
        )
        return completion.choices[0].message.content.strip()

    async def __call__(self, **kwargs) -> Any:
        query = kwargs.get("query")
        audio_path = kwargs.get("audio_path")

        if not query or not audio_path:
            return {
                "success": False, "output": None,
                "error": "Both query and audio_path are required.", "metrics": {},
            }

        if not os.path.isfile(audio_path):
            error_msg = f"Audio file not found: {audio_path}"
            logger.error(f"[ParseAudio] {error_msg}")
            return {"success": False, "output": None, "error": error_msg, "metrics": {}}

        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        ext = audio_path.rsplit(".", 1)[-1].lower() if "." in audio_path else "wav"
        logger.info(f"[ParseAudio] Processing audio: {audio_path} ({file_size_mb:.2f} MB, format: {ext})")

        # Get audio model config
        api_key, base_url, model = self._get_llm_config(self.AUDIO_MODEL)
        if not api_key:
            return {
                "success": False, "output": None,
                "error": "No API key found. Configure gpt-4o-audio-preview in model_config.yaml",
                "metrics": {},
            }

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        try:
            with open(audio_path, "rb") as f:
                raw_data = f.read()

            is_wav = raw_data[:4] == b'RIFF' and raw_data[8:12] == b'WAVE'

            if is_wav:
                duration = get_wav_duration(raw_data)
                logger.info(f"[ParseAudio] WAV duration: {duration:.1f}s")

                if duration <= MAX_SEGMENT_SECONDS:
                    # Short audio, send directly
                    audio_b64 = base64.b64encode(raw_data).decode("utf-8")
                    result = await self._transcribe_segment(
                        client, model, audio_b64, "wav", query,
                    )
                    logger.info(f"[ParseAudio] Short audio transcription succeeded, length: {len(result)}")
                    return {"success": True, "output": result, "error": None, "metrics": {}}

                else:
                    # Long audio, split into segments
                    segments = split_wav_segments(raw_data, MAX_SEGMENT_SECONDS)
                    logger.info(f"[ParseAudio] Long audio ({duration:.1f}s), split into {len(segments)} segments")

                    segment_results = []
                    for i, seg_data in enumerate(segments):
                        seg_b64 = base64.b64encode(seg_data).decode("utf-8")
                        seg_query = (
                            f"This is segment {i + 1} of {len(segments)} from a longer audio. "
                            f"Please transcribe this segment accurately."
                        )
                        try:
                            seg_text = await self._transcribe_segment(
                                client, model, seg_b64, "wav", seg_query,
                            )
                            segment_results.append(seg_text)
                            logger.info(f"[ParseAudio] Segment {i + 1}/{len(segments)} transcription succeeded")
                        except Exception as seg_exc:
                            logger.warning(
                                f"[ParseAudio] Segment {i + 1}/{len(segments)} transcription failed: {seg_exc}"
                            )
                            segment_results.append(
                                f"[Segment {i + 1} transcription failed: {seg_exc}]"
                            )

                    # Check if any segments succeeded
                    success_count = sum(
                        1 for r in segment_results if not r.startswith("[Segment")
                    )
                    if success_count == 0:
                        return {
                            "success": False, "output": None,
                            "error": "All audio segment transcriptions failed", "metrics": {},
                        }

                    # Use SUMMARY_MODEL to combine multi-segment results
                    combined = "\n\n".join(
                        f"[Segment {i + 1}]\n{text}"
                        for i, text in enumerate(segment_results)
                    )
                    summary_prompt = (
                        f"Below are transcriptions from multiple segments of the same audio file. "
                        f"Please combine them into a coherent complete transcription, "
                        f"then answer the user's question.\n\n"
                        f"## Segment Transcriptions:\n{combined}\n\n"
                        f"## User's Question:\n{query}\n\n"
                        f"Please provide the combined transcription first, then answer the question."
                    )
                    try:
                        sum_api_key, sum_base_url, sum_model = self._get_llm_config(
                            self.SUMMARY_MODEL
                        )
                        sum_client = AsyncOpenAI(
                            api_key=sum_api_key, base_url=sum_base_url,
                        )
                        sum_completion = await llm_retry(
                            sum_client.chat.completions.create,
                            model=sum_model,
                            messages=[{"role": "user", "content": summary_prompt}],
                            label=f"ParseAudio.summary({self.SUMMARY_MODEL})",
                        )
                        result = sum_completion.choices[0].message.content.strip()
                    except Exception as sum_exc:
                        logger.warning(
                            f"[ParseAudio] Summarization failed: {sum_exc}, returning concatenated results"
                        )
                        result = combined

                    logger.info(
                        f"[ParseAudio] Long audio processing completed, length: {len(result)}"
                    )
                    return {"success": True, "output": result, "error": None, "metrics": {}}

            else:
                # Non-WAV format (mp3, etc.)
                if file_size_mb > 3.0:
                    error_msg = (
                        f"Audio file too large ({file_size_mb:.1f} MB) and cannot be auto-segmented in non-WAV format. "
                        f"Please use WAV format or a smaller compressed file."
                    )
                    logger.error(f"[ParseAudio] {error_msg}")
                    return {
                        "success": False, "output": None,
                        "error": error_msg, "metrics": {},
                    }

                audio_b64 = base64.b64encode(raw_data).decode("utf-8")
                result = await self._transcribe_segment(
                    client, model, audio_b64, ext, query,
                )
                logger.info(
                    f"[ParseAudio] Non-WAV audio transcription succeeded, length: {len(result)}"
                )
                return {"success": True, "output": result, "error": None, "metrics": {}}

        except Exception as exc:
            error_msg = f"Audio processing failed: {type(exc).__name__}: {exc}"
            logger.error(f"[ParseAudio] {error_msg}")
            return {"success": False, "output": None, "error": error_msg, "metrics": {}}
