"""
PDDL generator with repair loop.  Exposes individual action schemas so
main.py can forward them to the coder.
"""
import time
# import logging
from pathlib import Path
from typing import Dict, Tuple, List, Set
from pydantic import BaseModel

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner, PlanValidator
from unified_planning.engines import PlanGenerationResultStatus

from llmclient import ChatGPTClient

MAX_RETRIES = 8
TMP_DOMAIN  = Path("domain.pddl")
TMP_PROBLEM = Path("problem.pddl")

SYSTEM_PROMPT = """
You are a classical-planning expert writing PDDL for an **OpenAI Gymnasium
MiniGrid** level.  
Return **only** a JSON object with exactly two keys:

    { "domain": "<complete PDDL DOMAIN file>",
      "problem": "<complete PDDL PROBLEM file>" }

No other text - no “```”, no explanations, no plan.

Guidelines
•  Model the high-level actions the Agent can call.
•  **Reuse existing method names** exactly when they already exist
   (move_forward, turn_left, pick_up, …).
•  If you invent a new high-level action, give it a clear, unique,
   snake_case name (e.g. cross_lava, move_to).
•  **Never reference raw (x,y) grid coordinates.**
   Describe motion and conditions with predicates such as
   (adjacent ?a ?b), (on ?o ?cell), (facing ?dir), etc.
   For example prefer  move_to(?obj)  over  move_to_xy.
•  Think at a human level of abstraction - the coder LLM will translate
   each new action into Python using numpy and the environment API.
•  The world contains walls, lava, doors, keys, boxes, balls, goals, etc.
   The agent must avoid obstacles and fulfill the mission.
•  All predicates, types and parameters must match between DOMAIN
   and PROBLEM.
"""

USER_PROMPT_TEMPLATE = """
Environment: {env_name}   (level “{level_name}”)
Category   : {category_name}
Skill      : {skill}

Level description:
{level_description}

Current Agent python code:
{agent_code}

Write DOMAIN and PROBLEM so that a plan exists using *only* the high-level
actions above (plus any brand-new actions you define following the
guidelines).  You are encouraged to invent whatever additional actions are
useful, as long as they obey the naming & abstraction rules.
"""

REFINEMENT_PROMPT_TEMPLATE = """
Planning / validation failed.

--- ERROR LOG ---
{error_log}

Please resend ONE JSON object (keys: domain, problem) that fixes the issue.
Do not include markdown or extra text.
""".strip()

class PDDLResp(BaseModel):
    domain: str
    problem: str

class PlannerLLM:
    def __init__(self):
        self.client = ChatGPTClient("o1", PDDLResp)
        # logging.basicConfig(level=logging.INFO,
        #                     format="%(levelname)s: %(message)s")
        
    def plan_with_unified_planning(self) -> Tuple:
        """
        Parse PDDL files, plan with Unified Planning, and validate the plan.
        Returns: (up_result, up_validation)
        """
        if not TMP_DOMAIN.exists() or not TMP_PROBLEM.exists():
            raise FileNotFoundError("PDDL files not found. "
                                    "Ensure 'domain.pddl' and 'problem.pddl' exist.")
        # Parse the PDDL files
        reader = PDDLReader()
        problem = reader.parse_problem("domain.pddl", "problem.pddl")
        
        with OneshotPlanner(problem_kind=problem.kind) as planner:
            result = planner.solve(problem)
        
        with PlanValidator(name="tamer") as validator:
            validation_result = validator.validate(problem, result.plan)
        
        return problem, result, validation_result
    

    def plan(self, meta: Dict):
        """
        meta keys: category_name, skill, level_name, level_description, env_name

        Returns: (problem_text, up_result, up_validation)
        """
        agent_src = Path(__file__).with_name("agent.py").read_text()

        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(
                env_name=meta["env_name"],
                level_name=meta["level_name"],
                category_name=meta["category_name"],
                skill=meta["skill"],
                level_description=meta["level_description"],
                agent_code=agent_src,
            )}
        ]

        # retry-repair loop
        for attempt in range(1, MAX_RETRIES + 1):
            resp: PDDLResp = self.client.chat_completion(conversation)
            dom_txt, prob_txt = resp.domain.strip(), resp.problem.strip()
            TMP_DOMAIN.write_text(dom_txt)
            TMP_PROBLEM.write_text(prob_txt)

            try:
                up_problem, up_result, up_validation = self.plan_with_unified_planning()
                if up_result.status in (
                        PlanGenerationResultStatus.SOLVED_SATISFICING,
                        PlanGenerationResultStatus.SOLVED_OPTIMALLY):
                    self._cache_action_schemas(dom_txt)
                    return prob_txt, up_result, up_validation
                err = f"Planner status: {up_result.status}"
            except Exception as exc:
                err = str(exc)

            # feed error back
            conversation.extend([
                {"role": "assistant", "content": resp.model_dump_json()},
                {"role": "user",      "content": REFINEMENT_PROMPT_TEMPLATE.format(
                    error_log=err
                )}
            ])
            time.sleep(1) # to avoid rate limits

        raise RuntimeError("PlannerLLM exhausted retries")

    # ------------ helpers ------------
    def _cache_action_schemas(self, domain_txt: str):
        """Build dict {action_name: full (:action …) block}."""
        self._schemas = {}
        lines = domain_txt.splitlines()
        buffer = []
        for ln in lines:
            if ln.lstrip().startswith("(:action"):
                buffer = [ln]
            elif buffer:
                buffer.append(ln)
                if ln.rstrip().endswith(")"):
                    name = buffer[0].split()[1]
                    self._schemas[name] = "\n".join(buffer)
                    buffer = []

    def get_action_schema(self, act_name: str) -> str:
        return self._schemas.get(act_name.replace("_", "-"), "")

    def get_action_schemas(self, act_names: Set[str]) -> Dict[str, str]:
        return {a: self.get_action_schema(a) for a in act_names}

