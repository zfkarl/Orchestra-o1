"""OmniGAIA benchmark - Video analysis tool.

Analyzes video content via multimodal LLMs by sampling frames from local video files.
When the video contains an audio track, the audio is automatically extracted and
transcribed, then combined with visual analysis for a comprehensive result.

Robustness:
- Automatically detects whether the video has an audio track (via ffprobe).
- If audio exists, extracts it via ffmpeg → transcribes with gpt-4o-audio-preview.
- If no audio track or ffmpeg is unavailable, gracefully falls back to visual-only analysis.
- Long audio (>30s) is automatically split into segments.
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from openai import AsyncOpenAI

from base.agent.base_action import BaseAction
from base.engine.async_llm import LLMsConfig, llm_retry
from base.engine.logs import logger
from benchmark.omnigaia.tools.audio_analysis import (
    get_wav_duration,
    split_wav_segments,
    MAX_SEGMENT_SECONDS,
)


# Module-level cache for ffmpeg/ffprobe binary paths
# (cannot be class attributes due to pydantic BaseModel restrictions)
_FFPROBE_PATH_CACHE: Optional[str] = None
_FFMPEG_PATH_CACHE: Optional[str] = None


class VideoAnalysisAction(BaseAction):
    """Analyze video content (visual frames + optional audio) via multimodal models.

    Processing pipeline:
    1. Extract frames at regular intervals → send to a vision LLM for visual analysis.
    2. Detect whether the video has an audio track (via ffprobe).
    3. If audio exists, extract it via ffmpeg → transcribe with gemini-3.1-pro-preview.
    4. If both visual and audio results are available, combine them into a
       comprehensive answer.

    Graceful degradation:
    - No audio track → returns visual-only result (no error).
    - ffmpeg/ffprobe not installed → returns visual-only result with a warning.
    - Audio transcription fails → returns visual-only result with a warning.
    - Frame extraction fails but audio works → returns audio-only result.
    """
    name: str = "VideoAnalysisAction"
    description: str = (
        "Analyze video content by extracting key frames AND audio track (if present). "
        "Frames are sent to a vision model for visual analysis; "
        "audio is automatically detected and transcribed via an audio model. "
        "Both results are combined for a comprehensive answer. "
        "Supports local video files (mp4, avi, mov, etc.)."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Instruction or question describing what to analyze in the video",
            },
            "video_path": {
                "type": "string",
                "description": "Local file path to the video file (e.g., mp4, avi, mov)",
            },
            "max_frames": {
                "type": "integer",
                "description": "Maximum number of frames to extract (default: 8, max: 16)",
            },
            "start_time": {
                "type": "number",
                "description": "Start time in seconds for frame extraction (default: 0)",
            },
            "end_time": {
                "type": "number",
                "description": (
                    "End time in seconds for frame extraction (default: video duration). "
                    "Use this with start_time to focus on a specific segment."
                ),
            },
            "analyze_audio": {
                "type": "boolean",
                "description": (
                    "Whether to also extract and transcribe the audio track (default: true). "
                    "Set to false to skip audio analysis for faster processing when "
                    "only visual information is needed."
                ),
            },
        },
        "required": ["query", "video_path"],
        "additionalProperties": False,
    }

    # Model for visual frame analysis (must support vision / image_url)
    VISION_MODEL: str = "gpt-5"
    # Model for audio transcription (must support input_audio modality)
    AUDIO_MODEL: str = "gpt-4o-audio-preview"
    # Model for combining visual + audio results into a final answer
    SUMMARY_MODEL: str = "gpt-5"

    # ------------------------------------------------------------------ #
    #  ffmpeg / ffprobe binary discovery
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_binary(name: str) -> Optional[str]:
        """Find ffmpeg or ffprobe binary with multi-strategy lookup.

        Search order:
        1. System PATH (shutil.which)
        2. Same directory as the current Python interpreter (conda env bin/)
        3. Common conda env locations
        4. /usr/bin, /usr/local/bin

        Returns the full path to the binary, or None if not found.
        """
        # Strategy 1: system PATH
        found = shutil.which(name)
        if found:
            return found

        # Strategy 2: same directory as the Python interpreter (covers conda envs)
        python_bin_dir = Path(sys.executable).resolve().parent
        candidate = python_bin_dir / name
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            return str(candidate)

        # Strategy 3: common locations
        for search_dir in ["/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"]:
            candidate = Path(search_dir) / name
            if candidate.exists() and os.access(str(candidate), os.X_OK):
                return str(candidate)

        return None

    @staticmethod
    def _get_ffprobe() -> Optional[str]:
        """Get the path to ffprobe binary (cached at module level)."""
        global _FFPROBE_PATH_CACHE
        if _FFPROBE_PATH_CACHE is None:
            found = VideoAnalysisAction._find_binary("ffprobe") or ""
            _FFPROBE_PATH_CACHE = found
            if found:
                logger.info(f"[VideoAnalysis] Found ffprobe at: {found}")
            else:
                logger.warning("[VideoAnalysis] ffprobe not found in any known location")
        return _FFPROBE_PATH_CACHE or None

    @staticmethod
    def _get_ffmpeg() -> Optional[str]:
        """Get the path to ffmpeg binary (cached at module level)."""
        global _FFMPEG_PATH_CACHE
        if _FFMPEG_PATH_CACHE is None:
            found = VideoAnalysisAction._find_binary("ffmpeg") or ""
            _FFMPEG_PATH_CACHE = found
            if found:
                logger.info(f"[VideoAnalysis] Found ffmpeg at: {found}")
            else:
                logger.warning("[VideoAnalysis] ffmpeg not found in any known location")
        return _FFMPEG_PATH_CACHE or None

    # ------------------------------------------------------------------ #
    #  LLM config helper
    # ------------------------------------------------------------------ #

    def _get_llm_config(self, model_name: Optional[str] = None) -> tuple:
        """Get LLM configuration from model_config.yaml or environment variables."""
        model_name = model_name or self.VISION_MODEL
        try:
            llms_config = LLMsConfig.default()
            model_config = llms_config.get(model_name)
            if model_config:
                return (model_config.key, model_config.base_url, model_config.model)
        except Exception:
            pass
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        base_url = (
            os.getenv("LLM_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://newapi.deepwisdom.ai/v1"
        )
        return (api_key, base_url, model_name)

    # ------------------------------------------------------------------ #
    #  Video duration / frame extraction (visual)
    # ------------------------------------------------------------------ #

    def _get_video_duration(self, video_path: str) -> Optional[float]:
        """Get video duration in seconds using OpenCV."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"[VideoAnalysis] Cannot open video: {video_path}")
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and frame_count > 0:
                return frame_count / fps
            return None
        except Exception as e:
            logger.warning(f"[VideoAnalysis] Failed to get video duration: {e}")
            return None

    def _extract_frames(
        self,
        video_path: str,
        max_frames: int = 8,
        start_time: float = 0,
        end_time: Optional[float] = None,
    ) -> List[str]:
        """Extract frames from video using OpenCV; return base64-encoded JPEG images."""
        duration = self._get_video_duration(video_path)
        if duration is None:
            duration = 300  # fallback: assume 5 minutes

        if end_time is None or end_time > duration:
            end_time = duration
        if start_time < 0:
            start_time = 0
        if start_time >= end_time:
            start_time = max(0, end_time - 10)

        segment_duration = end_time - start_time
        if segment_duration <= 0:
            interval = 1.0
        else:
            interval = max(1.0, segment_duration / max_frames)

        frames: List[str] = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"[VideoAnalysis] Cannot open video: {video_path}")
                return []

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0

            start_frame = int(start_time * fps)
            frame_interval = max(1, int(interval * fps))
            end_frame = int(end_time * fps)
            current_frame = start_frame

            while len(frames) < max_frames and current_frame < end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                if not ret:
                    break
                success, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if success:
                    frames.append(base64.b64encode(buffer.tobytes()).decode("utf-8"))
                current_frame += frame_interval

            cap.release()
        except Exception as e:
            logger.error(f"[VideoAnalysis] Frame extraction failed: {e}")
            return []

        return frames

    # ------------------------------------------------------------------ #
    #  Audio track detection and extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_has_audio_track(video_path: str) -> bool:
        """Use ffprobe to check whether the video file contains an audio stream.

        Returns False if ffprobe is not available or the video has no audio.
        """
        ffprobe = VideoAnalysisAction._get_ffprobe()
        if not ffprobe:
            logger.warning(
                "[VideoAnalysis] ffprobe not found; cannot detect audio track. "
                "Install ffmpeg for full video+audio analysis."
            )
            return False

        try:
            result = subprocess.run(
                [
                    ffprobe, "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    video_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # ffprobe prints "audio" for each audio stream found
            return "audio" in result.stdout
        except FileNotFoundError:
            logger.warning(
                "[VideoAnalysis] ffprobe not found; cannot detect audio track. "
                "Install ffmpeg for full video+audio analysis."
            )
            return False
        except subprocess.TimeoutExpired:
            logger.warning("[VideoAnalysis] ffprobe timed out while checking audio track")
            return False
        except Exception as e:
            logger.warning(f"[VideoAnalysis] ffprobe check failed: {e}")
            return False

    def _extract_audio_from_video(
        self,
        video_path: str,
        start_time: float = 0,
        end_time: Optional[float] = None,
    ) -> Optional[str]:
        """Extract audio track from video file using ffmpeg.

        Returns the path to a temporary WAV file, or None if extraction fails
        (e.g. no audio track, ffmpeg unavailable, corrupted stream).
        The caller is responsible for deleting the temp file.
        """
        tmp_wav_path: Optional[str] = None
        try:
            tmp_fd, tmp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(tmp_fd)

            ffmpeg = self._get_ffmpeg()
            if not ffmpeg:
                logger.warning(
                    "[VideoAnalysis] ffmpeg not found; cannot extract audio. "
                    "Install ffmpeg for full video+audio analysis."
                )
                self._safe_unlink(tmp_wav_path)
                return None

            cmd = [ffmpeg, "-y", "-i", video_path]
            if start_time > 0:
                cmd.extend(["-ss", str(start_time)])
            if end_time is not None:
                cmd.extend(["-t", str(end_time - start_time)])
            cmd.extend([
                "-vn",                  # discard video
                "-acodec", "pcm_s16le", # 16-bit PCM
                "-ar", "16000",         # 16 kHz (good for speech)
                "-ac", "1",             # mono
                tmp_wav_path,
            ])

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )

            if result.returncode != 0:
                logger.warning(
                    f"[VideoAnalysis] ffmpeg audio extraction failed "
                    f"(exit {result.returncode}): {result.stderr[:300]}"
                )
                self._safe_unlink(tmp_wav_path)
                return None

            file_size = os.path.getsize(tmp_wav_path)
            if file_size < 100:
                logger.info(
                    "[VideoAnalysis] Extracted audio file is too small "
                    f"({file_size} bytes), video likely has no audible content"
                )
                self._safe_unlink(tmp_wav_path)
                return None

            logger.info(
                f"[VideoAnalysis] Audio extracted to temp file "
                f"({file_size / 1024:.1f} KB)"
            )
            return tmp_wav_path

        except FileNotFoundError:
            logger.warning(
                "[VideoAnalysis] ffmpeg not found; cannot extract audio. "
                "Install ffmpeg for full video+audio analysis."
            )
            self._safe_unlink(tmp_wav_path)
            return None
        except subprocess.TimeoutExpired:
            logger.warning("[VideoAnalysis] ffmpeg audio extraction timed out (>120s)")
            self._safe_unlink(tmp_wav_path)
            return None
        except Exception as e:
            logger.warning(f"[VideoAnalysis] Audio extraction error: {e}")
            self._safe_unlink(tmp_wav_path)
            return None

    @staticmethod
    def _safe_unlink(path: Optional[str]) -> None:
        """Delete a file if it exists, ignoring errors."""
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    #  Audio transcription (reuses WAV helpers from audio_analysis)
    # ------------------------------------------------------------------ #

    async def _transcribe_audio(self, wav_path: str, query: str) -> Optional[str]:
        """Transcribe extracted WAV audio using gemini-3.1-pro-preview.

        Long audio is automatically split into ≤30 s segments.
        Returns None on failure.
        """
        api_key, base_url, model = self._get_llm_config(self.AUDIO_MODEL)
        if not api_key:
            logger.warning(
                "[VideoAnalysis] No API key for audio model; skipping audio analysis"
            )
            return None

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        try:
            with open(wav_path, "rb") as f:
                raw_data = f.read()

            duration = get_wav_duration(raw_data)
            logger.info(f"[VideoAnalysis] Audio track duration: {duration:.1f}s")

            if duration < 0.5:
                logger.info("[VideoAnalysis] Audio track too short (<0.5s), skipping")
                return None

            if duration <= MAX_SEGMENT_SECONDS:
                # Short audio — send in one request
                audio_b64 = base64.b64encode(raw_data).decode("utf-8")
                completion = await llm_retry(
                    client.chat.completions.create,
                    model=model,
                    #modalities=["text"],
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This audio was extracted from a video. "
                                    "Please transcribe it and describe any notable sounds. "
                                    f"Context: {query}"
                                ),
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_b64, "format": "wav"},
                            },
                        ],
                    }],
                    label=f"VideoAnalysis._transcribe_audio(short, {model})",
                )
                result = completion.choices[0].message.content
                return result.strip() if result else None

            else:
                # Long audio — split into segments
                segments = split_wav_segments(raw_data, MAX_SEGMENT_SECONDS)
                logger.info(
                    f"[VideoAnalysis] Audio split into {len(segments)} segments "
                    f"(total {duration:.1f}s)"
                )

                segment_results: List[str] = []
                for i, seg_data in enumerate(segments):
                    seg_b64 = base64.b64encode(seg_data).decode("utf-8")
                    seg_query = (
                        f"This is audio segment {i + 1} of {len(segments)} "
                        f"extracted from a video. Please transcribe it accurately."
                    )
                    try:
                        completion = await llm_retry(
                            client.chat.completions.create,
                            model=model,
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": seg_query},
                                    {
                                        "type": "input_audio",
                                        "input_audio": {
                                            "data": seg_b64, "format": "wav",
                                        },
                                    },
                                ],
                            }],
                            label=f"VideoAnalysis._transcribe_audio(seg {i+1}/{len(segments)}, {model})",
                        )
                        seg_text = completion.choices[0].message.content
                        if seg_text:
                            segment_results.append(seg_text.strip())
                        logger.info(
                            f"[VideoAnalysis] Audio segment {i + 1}/{len(segments)} "
                            f"transcribed"
                        )
                    except Exception as seg_exc:
                        logger.warning(
                            f"[VideoAnalysis] Audio segment {i + 1}/{len(segments)} "
                            f"failed: {seg_exc}"
                        )
                        segment_results.append(
                            f"[Segment {i + 1} transcription failed]"
                        )

                # Check if any segments succeeded
                successful = [
                    r for r in segment_results if not r.startswith("[Segment")
                ]
                if not successful:
                    logger.warning(
                        "[VideoAnalysis] All audio segment transcriptions failed"
                    )
                    return None

                return "\n\n".join(
                    f"[Audio Segment {i + 1}] {text}"
                    for i, text in enumerate(segment_results)
                )

        except Exception as e:
            logger.warning(f"[VideoAnalysis] Audio transcription failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Combine visual + audio analysis
    # ------------------------------------------------------------------ #

    async def _combine_visual_and_audio(
        self,
        visual_result: str,
        audio_result: str,
        query: str,
    ) -> str:
        """Use SUMMARY_MODEL to combine visual analysis and audio transcription
        into a single comprehensive answer."""
        api_key, base_url, model = self._get_llm_config(self.SUMMARY_MODEL)
        if not api_key:
            # Fallback: concatenate both results
            return (
                f"[Visual Analysis]\n{visual_result}\n\n"
                f"[Audio Transcription]\n{audio_result}"
            )

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        prompt = (
            "I analyzed a video that contains both visual frames and an audio track. "
            "Please combine the following analyses to provide a comprehensive answer.\n\n"
            f"## Visual Analysis (from video frames):\n{visual_result}\n\n"
            f"## Audio Transcription (from audio track):\n{audio_result}\n\n"
            f"## User's Question:\n{query}\n\n"
            "Please provide a comprehensive answer that integrates both "
            "visual and audio information."
        )

        try:
            completion = await llm_retry(
                client.chat.completions.create,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                label=f"VideoAnalysis._combine_visual_and_audio({model})",
            )
            result = completion.choices[0].message.content
            return result.strip() if result else (
                f"[Visual Analysis]\n{visual_result}\n\n"
                f"[Audio Transcription]\n{audio_result}"
            )
        except Exception as e:
            logger.warning(
                f"[VideoAnalysis] Combination via LLM failed ({e}); "
                f"returning separate results"
            )
            return (
                f"[Visual Analysis]\n{visual_result}\n\n"
                f"[Audio Transcription]\n{audio_result}"
            )

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #

    async def __call__(self, **kwargs) -> Any:
        query = kwargs.get("query")
        video_path = kwargs.get("video_path")
        max_frames = min(int(kwargs.get("max_frames", 8)), 16)
        start_time = float(kwargs.get("start_time", 0))
        end_time = kwargs.get("end_time")
        analyze_audio = kwargs.get("analyze_audio", True)
        if end_time is not None:
            end_time = float(end_time)

        # --- Parameter validation ---
        if not query or not video_path:
            return {
                "success": False, "output": None,
                "error": "Both query and video_path are required.", "metrics": {},
            }

        if not Path(video_path).exists():
            return {
                "success": False, "output": None,
                "error": f"Video file not found: {video_path}", "metrics": {},
            }

        api_key, base_url, model = self._get_llm_config(self.VISION_MODEL)
        if not api_key:
            return {
                "success": False, "output": None,
                "error": (
                    "No API key found. Configure in model_config.yaml "
                    "or set OPENAI_API_KEY env variable."
                ),
                "metrics": {},
            }

        # ===== Step 1: Visual analysis (extract frames) =====
        logger.info(
            f"[VideoAnalysis] Extracting up to {max_frames} frames "
            f"from {video_path}"
        )
        frames = self._extract_frames(video_path, max_frames, start_time, end_time)

        visual_result: Optional[str] = None
        if frames:
            logger.info(
                f"[VideoAnalysis] Extracted {len(frames)} frames, "
                f"sending to vision model ({model})..."
            )

            content: List[Dict[str, Any]] = []
            time_note = ""
            if start_time > 0 or end_time is not None:
                time_note = (
                    f" (analyzing segment from {start_time}s "
                    f"to {end_time if end_time else 'end'}s)"
                )

            content.append({
                "type": "text",
                "text": (
                    f"The following {len(frames)} frames are extracted "
                    f"from a video{time_note}. {query}"
                ),
            })
            for frame_b64 in frames:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                })

            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            try:
                completion = await llm_retry(
                    client.chat.completions.create,
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    label=f"VideoAnalysis.visual({model})",
                )
                result_text = completion.choices[0].message.content
                if result_text:
                    visual_result = result_text.strip()
                    logger.info(
                        f"[VideoAnalysis] Visual analysis completed "
                        f"({len(visual_result)} chars)"
                    )
            except Exception as exc:
                logger.warning(f"[VideoAnalysis] Visual analysis failed: {exc}")
        else:
            logger.warning("[VideoAnalysis] No frames extracted from video")

        # ===== Step 2: Audio analysis (only if requested) =====
        audio_result: Optional[str] = None
        has_audio_track = False
        tmp_wav_path: Optional[str] = None

        if analyze_audio:
            # First, check if the video actually has an audio track
            has_audio_track = self._check_has_audio_track(video_path)

            if has_audio_track:
                logger.info(
                    "[VideoAnalysis] Audio track detected, extracting..."
                )
                try:
                    tmp_wav_path = self._extract_audio_from_video(
                        video_path, start_time, end_time,
                    )
                    if tmp_wav_path:
                        audio_result = await self._transcribe_audio(
                            tmp_wav_path, query,
                        )
                        if audio_result:
                            logger.info(
                                f"[VideoAnalysis] Audio transcription completed "
                                f"({len(audio_result)} chars)"
                            )
                        else:
                            logger.info(
                                "[VideoAnalysis] Audio transcription returned "
                                "empty result"
                            )
                finally:
                    # Always clean up temp file
                    self._safe_unlink(tmp_wav_path)
            else:
                logger.info(
                    "[VideoAnalysis] No audio track detected in video; "
                    "proceeding with visual-only analysis"
                )
        else:
            logger.info(
                "[VideoAnalysis] Audio analysis disabled by caller "
                "(analyze_audio=false)"
            )

        # ===== Step 3: Combine results =====
        if visual_result and audio_result:
            # Both visual and audio available — combine for best answer
            logger.info("[VideoAnalysis] Combining visual + audio analyses...")
            combined = await self._combine_visual_and_audio(
                visual_result, audio_result, query,
            )
            return {
                "success": True,
                "output": combined,
                "error": None,
                "metrics": {
                    "frames_analyzed": len(frames),
                    "has_audio_track": True,
                    "audio_transcribed": True,
                    "analysis_type": "visual+audio",
                },
            }

        elif visual_result:
            # Visual only (no audio track, or audio extraction/transcription failed)
            return {
                "success": True,
                "output": visual_result,
                "error": None,
                "metrics": {
                    "frames_analyzed": len(frames),
                    "has_audio_track": has_audio_track,
                    "audio_transcribed": False,
                    "analysis_type": "visual_only",
                },
            }

        elif audio_result:
            # Audio only (frame extraction failed but audio worked — rare)
            return {
                "success": True,
                "output": audio_result,
                "error": None,
                "metrics": {
                    "frames_analyzed": 0,
                    "has_audio_track": True,
                    "audio_transcribed": True,
                    "analysis_type": "audio_only",
                },
            }

        else:
            # Both failed
            return {
                "success": False,
                "output": None,
                "error": (
                    "Failed to extract both frames and audio from the video. "
                    "Ensure OpenCV and ffmpeg are installed, and the video file "
                    "is valid."
                ),
                "metrics": {},
            }
