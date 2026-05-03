"""
Trace Formatter - Abstracted trace formatting utility

Provides extensible trace formatting interface supporting different benchmark action/observation formats.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Protocol, Callable


class StepLike(Protocol):
    """Step protocol, compatible with StepRecord structure"""
    action: Dict[str, Any]
    observation: Any
    reward: float
    done: bool
    info: Dict[str, Any]


class ActionFormatter(ABC):
    """Action formatter base class"""
    
    @property
    @abstractmethod
    def action_type(self) -> str:
        """Return the action type this formatter handles"""
        ...
    
    @abstractmethod
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        """Format action to readable string"""
        ...


class ObservationFormatter(ABC):
    """Observation formatter base class"""
    
    @abstractmethod
    def can_format(self, obs: Dict[str, Any]) -> bool:
        """Check if this formatter can handle the observation"""
        ...
    
    @abstractmethod
    def format(self, obs: Dict[str, Any], max_len: int = 300) -> tuple[str, str]:
        """
        Format observation
        
        Returns:
            tuple[str, str]: (status line, output content)
        """
        ...


class FinishActionFormatter(ActionFormatter):
    """finish action formatter (generic)"""
    
    @property
    def action_type(self) -> str:
        return "finish"
    
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        status = params.get("status", "done")
        msg = params.get("message", "")[:60]
        return f'finish(status="{status}", msg="{msg}")'


# ============== OmniGAIA Formatters ==============

class GoogleSearchActionFormatter(ActionFormatter):
    """OmniGAIA GoogleSearchAction formatter"""
    
    @property
    def action_type(self) -> str:
        return "GoogleSearchAction"
    
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        query = params.get("query", "")[:80]
        return f'GoogleSearch(query="{query}")'


class ExtractUrlActionFormatter(ActionFormatter):
    """OmniGAIA ExtractUrlContentAction formatter"""
    
    @property
    def action_type(self) -> str:
        return "ExtractUrlContentAction"
    
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        url = params.get("url", "")[:60]
        browse_query = params.get("browse_query", "")[:40]
        return f'ExtractUrl(url="{url}", query="{browse_query}")'


class ExecuteCodeActionFormatter(ActionFormatter):
    """OmniGAIA ExecuteCodeAction formatter"""
    
    @property
    def action_type(self) -> str:
        return "ExecuteCodeAction"
    
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        code = params.get("code", "")[:80].replace("\n", " ")
        return f'ExecuteCode(code="{code}...")'


class ImageAnalysisActionFormatter(ActionFormatter):
    """OmniGAIA ImageAnalysisAction formatter"""
    
    @property
    def action_type(self) -> str:
        return "ImageAnalysisAction"
    
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        query = params.get("query", "")[:60]
        image_path = params.get("image_path", "")[:40]
        return f'ImageAnalysis(query="{query}", image="{image_path}")'


class VideoAnalysisActionFormatter(ActionFormatter):
    """OmniGAIA VideoAnalysisAction formatter"""
    
    @property
    def action_type(self) -> str:
        return "VideoAnalysisAction"
    
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        query = params.get("query", "")[:60]
        video_path = params.get("video_path", "")[:40]
        max_frames = params.get("max_frames", 8)
        analyze_audio = params.get("analyze_audio", True)
        return f'VideoAnalysis(query="{query}", video="{video_path}", frames={max_frames}, audio={analyze_audio})'


class ParseAudioActionFormatter(ActionFormatter):
    """OmniGAIA ParseAudioAction formatter"""
    
    @property
    def action_type(self) -> str:
        return "ParseAudioAction"
    
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        query = params.get("query", "")[:60]
        audio_path = params.get("audio_path", "")[:40]
        return f'ParseAudio(query="{query}", audio="{audio_path}")'


class SuccessObservationFormatter(ObservationFormatter):
    """OmniGAIA observation formatter (success + output/error)"""
    
    def can_format(self, obs: Dict[str, Any]) -> bool:
        return "success" in obs
    
    def format(self, obs: Dict[str, Any], max_len: int = 300) -> tuple[str, str]:
        success = obs.get("success", False)
        output = str(obs.get("output", obs.get("error", "")))
        return f"success={success}", output


# ============== Fallback Formatters ==============

class FallbackActionFormatter(ActionFormatter):
    """Generic fallback action formatter"""
    
    def __init__(self, action_type: str = "unknown"):
        self._action_type = action_type
    
    @property
    def action_type(self) -> str:
        return self._action_type
    
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        param_keys = list(params.keys())[:3]
        return f'{self._action_type}({param_keys})'


class FallbackObservationFormatter(ObservationFormatter):
    """Generic fallback observation formatter"""
    
    def can_format(self, obs: Dict[str, Any]) -> bool:
        return True  # Always handles
    
    def format(self, obs: Dict[str, Any], max_len: int = 300) -> tuple[str, str]:
        return "", str(obs)


# ============== TraceFormatter Main Class ==============

class TraceFormatter:
    """
    Trace formatter
    
    Implements extensible formatting logic via registered ActionFormatter and ObservationFormatter.
    """
    
    def __init__(self):
        self._action_formatters: Dict[str, ActionFormatter] = {}
        self._obs_formatters: List[ObservationFormatter] = []
        self._fallback_obs_formatter = FallbackObservationFormatter()
    
    def register_action_formatter(self, formatter: ActionFormatter) -> "TraceFormatter":
        """Register action formatter"""
        self._action_formatters[formatter.action_type] = formatter
        return self
    
    def register_obs_formatter(self, formatter: ObservationFormatter) -> "TraceFormatter":
        """Register observation formatter"""
        self._obs_formatters.append(formatter)
        return self
    
    def format_action(self, action: Dict[str, Any], max_len: int = 100) -> str:
        """Format single action"""
        action_type = action.get("action", "unknown")
        params = action.get("params", {})
        
        formatter = self._action_formatters.get(action_type)
        if formatter:
            return formatter.format(params, max_len)
        return FallbackActionFormatter(action_type).format(params, max_len)
    
    def format_observation(self, obs: Any, max_len: int = 300) -> tuple[str, str]:
        """Format single observation"""
        if not isinstance(obs, dict):
            return "", str(obs)
        
        for formatter in self._obs_formatters:
            if formatter.can_format(obs):
                return formatter.format(obs, max_len)
        
        return self._fallback_obs_formatter.format(obs, max_len)
    
    def format_trace(self, trace: List[StepLike], max_output_len: int = 300) -> str:
        """
        Format complete trace
        
        Args:
            trace: List of steps
            max_output_len: Output truncation length
        
        Returns:
            str: Formatted trace text
        """
        if not trace:
            return "No steps executed"
        
        lines = []
        for i, step in enumerate(trace, 1):
            # Format action
            action_str = self.format_action(step.action)
            lines.append(f"Step {i}: {action_str}")
            
            # Format observation
            status_line, output = self.format_observation(step.observation, max_output_len)
            if status_line:
                lines.append(f"  → {status_line}")
            
            # Truncate output
            if len(output) > max_output_len:
                output = output[:max_output_len] + f"...[+{len(output)-max_output_len} chars]"
            output = output.replace("\n", " ").strip()
            lines.append(f"  → output: {output}")
            lines.append("")
        
        return "\n".join(lines)


# ============== Pre-built Formatter Factories ==============

def create_gaia_formatter() -> TraceFormatter:
    """Create OmniGAIA formatter"""
    return (
        TraceFormatter()
        .register_action_formatter(GoogleSearchActionFormatter())
        .register_action_formatter(ExtractUrlActionFormatter())
        .register_action_formatter(ExecuteCodeActionFormatter())
        .register_action_formatter(ImageAnalysisActionFormatter())
        .register_action_formatter(VideoAnalysisActionFormatter())
        .register_action_formatter(ParseAudioActionFormatter())
        .register_action_formatter(FinishActionFormatter())
        .register_obs_formatter(SuccessObservationFormatter())
    )

