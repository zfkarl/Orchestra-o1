"""
Backward compatible import proxy

OrchestraSubAgent has been moved to orchestra_o1.subagents.react_agent.ReActAgent
This file maintains backward compatibility.
"""
from orchestra_o1.subagents.react_agent import ReActAgent

# Backward compatibility alias
OrchestraSubAgent = ReActAgent

__all__ = ["OrchestraSubAgent", "ReActAgent"]
