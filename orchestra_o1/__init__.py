"""
Orchestra-o1 — Unified orchestration framework for the OmniGAIA benchmark.

Provides a MainAgent + SubAgent architecture where the MainAgent delegates
multimodal tasks to SubAgents for execution.
"""
from orchestra_o1.subagents import ReActAgent
from orchestra_o1.sub_agent import OrchestraSubAgent  # Backward compatibility
from orchestra_o1.config import GAIAOrchestraConfig

__all__ = [
    # SubAgents
    "ReActAgent",
    "OrchestraSubAgent",  # Backward compatibility alias
    # Configs
    "GAIAOrchestraConfig",
]
