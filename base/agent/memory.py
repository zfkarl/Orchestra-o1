from __future__ import annotations

from abc import ABC
from typing import List, Dict, Any, Optional

import json

from base.engine.async_llm import AsyncLLM 
from benchmark.common.env import Observation, Action


class Memory(ABC):

    def __init__(self, llm: AsyncLLM, max_memory: int = 10, keep_recent: int = 3):
        self.llm = llm
        self.max_memory = max_memory
        self.keep_recent = keep_recent
        self._records: List[Dict[str, Any]] = []
        self._summary: Optional[str] = None

    async def add_memory(
        self,
        obs: Observation,
        action: Action,
        thinking: Optional[str] = None,
        reward: Optional[float] = None,
        raw_response: Optional[str] = None,
    ) -> None:
        """
        Add a new memory record, then trigger compression if needed.
        """
        record = {
            "observation": obs,
            "action": action,
            "thinking": thinking,
            "reward": reward,
            "raw_response": raw_response,
        }
        self._records.append(record)
        await self._compress()

    def _get_text(self) -> str:
        """
        Return readable memory text for prompts.
        """
        if not self._summary and not self._records:
            return "None"

        parts: List[str] = []

        if self._records:
            parts.append("[Recent steps (latest first)]")
            for idx, r in enumerate(reversed(self._records), 1):
                act = r.get("action")
                obs = r.get("observation")
                thinking = r.get("thinking")
                reward = r.get("reward")
                parts.append(f"{idx}. act={act}, obs={obs}, thinking={thinking}, reward={reward}")

        if self._summary:
            parts.append("")
            parts.append("[Summary of earlier steps]")
            parts.append(self._summary)

        return "\n".join(parts)

    def as_text(self) -> str:
        return self._get_text()

    async def _compress(self) -> None:
        """
        When record count reaches max_memory:
        - Compress older records into the summary.
        - Keep only the most recent keep_recent raw records.
        """
        if len(self._records) < self.max_memory:
            return

        if self.keep_recent > 0:
            head = self._records[:-self.keep_recent]
            tail = self._records[-self.keep_recent:]
        else:
            head = self._records[:]
            tail = []

        if head:
            head_summary = await self._summarize_records(head)
            if self._summary:
                # Append to existing summary for continuity
                self._summary += "\n\n" + head_summary
            else:
                self._summary = head_summary
                
        self._records = tail

    async def _summarize_records(self, records: List[Dict[str, Any]]) -> str:
        """
        Compress a batch of records using the LLM.
        """

        record_lines: List[str] = []
        for idx, r in enumerate(records, 1):
            act = r.get("action")
            obs = r.get("observation")
            reward = r.get("reward")
            thinking = r.get("thinking")
            record_lines.append(
                f"{idx}. action={json.dumps(act, ensure_ascii=False)}, "
                f"observation={json.dumps(obs, ensure_ascii=False)}, "
                f"thinking={json.dumps(thinking, ensure_ascii=False)}, "
                f"reward={reward}"
            )
        records_text = "\n".join(record_lines)

        summary_prompt = f"""
You are the memory compression module of a language-model-based agent.

You are given several past interaction steps in chronological order (oldest first).
Each step includes:
- the agent's action,
- the environment observation,
- the reward signal.
- the agent's thinking.

Your task is to write a compact, **persistent memory** block that lets the agent
continue its work without seeing the full history.

Please:
- Focus on:
  1) stable facts and rules about the environment/world,
  2) useful strategies / plans / tools the agent tried,
  3) important mistakes or failure patterns to avoid later,
  4) partial progress and remaining goals / TODOs.
- Use at most 8–10 lines.
- Use a neutral, factual tone.
- Do NOT repeat low-level JSON details unless they are crucial.
- Do NOT include meta text like "here is the summary" or any explanation.
- Output ONLY the memory lines, one per line.

===== PAST STEPS =====
{records_text}
===== SUMMARY (start here) =====
""".strip()

        resp = await self.llm(summary_prompt)

        return resp

    def clear(self) -> None:
        """
        Clear all memories and summary.
        """
        self._records.clear()
        self._summary = None
