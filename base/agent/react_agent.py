from pydantic import Field
from typing import Dict, Tuple

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory
from base.engine.utils import parse_llm_action_response
from base.engine.logs import logger
from benchmark.common.env import BasicInfo, Observation, Action


REACT_PROMPT = """
==== Instruction ====
{instruction}

==== Action Space ====
{action_space}

==== Output ====

==== Thinking ====
You should think step by step before you output an action. 

==== Action Output Format ====
When you output the action, 
you should output the action name and parameters in the JSON format, and only one action.
Such as, 
```json
{{
    "action": "",
    "params": {{
        "<param_name>": "<param_value>"
    }}
}}
```
==== Memory ====
Recent memory:
{memory}

==== Observation ====
{obs}
"""

class ReAcTAgent(BaseAgent):
    """
    A Basic ReAcT Agent. 
    """
    name: str = Field(default="ReAcTAgent")
    description: str = Field(default="A Basic ReAcT Agent.")
    current_env_instruction: str = Field(default="")
    current_action_space: str = Field(default="")
    trajectory_folder_path: str = Field(default="")
    memory: Memory = Field(default=None)

    def reset(self, env_info: "BasicInfo") -> None:
        self.memory = Memory(llm=self.llm, max_memory=10)
        self.current_env_instruction = env_info.instruction
        self.current_action_space = env_info.action_space
        self.memory.clear()

    def parse_action(self, resp: str):
        """Parse LLM response to extract action data."""
        return parse_llm_action_response(resp)

    def _get_memory(self) -> str:
        return self.memory.as_text()

    def _get_max_steps(self, env, env_info: Dict) -> int:
        explicit = env_info.get("max_step")
        if explicit is not None:
            try:
                return int(explicit)
            except Exception:
                pass
        configs = getattr(env, "configs", {}) or {}
        term_steps = configs.get("termination", {}).get("max_steps")
        try:
            if term_steps is not None:
                return int(term_steps)
        except Exception:
            pass
        return 20

    async def step(self, agent_obs: Observation) -> Tuple[Action, str, str]:        
        act_prompt = REACT_PROMPT.format(
            instruction = self.current_env_instruction,
            action_space = self.current_action_space,
            obs = agent_obs,
            memory = self._get_memory()
        )
        logger.info(f"Agent Input:\n{act_prompt}")
        try:
            resp = await self.llm(act_prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            resp = ""

        action = self.parse_action(resp)
        logger.agent_action(f"Agent Action: {action}")

        await self.memory.add_memory(obs=agent_obs, action=action, raw_response=resp)
        return action, resp, act_prompt

    async def run(self):
        pass

