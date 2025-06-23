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
You are a classical-planning expert for an **OpenAI Gymnasium MiniGrid** agent.

Return one JSON object only:

    { "domain": "<full PDDL DOMAIN>",
      "problem": "<full PDDL PROBLEM>" }

No markdown, no plan, no commentary.

Abstraction rules
•  Stay semantic - don't enumerate every grid cell or (x,y) coordinate.
•  You still need a **minimal type system** (e.g. agent, target) and at
   least one constant of each; otherwise the PDDL won't parse.
•  State is expressed only through high-level predicates about the agent
   and named objects (goal, key1, door1, …), e.g.
        (at_goal) (holding ?o) (door_open ?d) (safe) …
•  The Agent already supports: move_forward, turn_left, turn_right,
   pick_up, drop, toggle, done, safe_forward, pick_up_obj.
•  From the **Environment / Category / Skill / Level description**, decide 
   whether the **currently available high-level actions are sufficient**; 
   reuse them only if they can solve the mission exactly. 
   If none fully fit, invent one or more new *snake_case* actions that do,
   that the coder will later implement (e.g. cross_lava, move_to_goal).
•  Keep DOMAIN compact - a handful of predicates and actions.
•  All predicate / parameter names must match between DOMAIN and PROBLEM.
•  If several versions of an action exist (e.g. move_to_goal, move_to_goal_v2, move_to_goal_v3), 
   always reference the highest-numbered suffix currently present in the Agent code.

Syntax constraints (very important)
•  **Do NOT use `(not …)`** in preconditions/effects unless you also add
   `:negative-preconditions` to `:requirements`.  Simpler: just avoid `not`.
•  Do **not** include comments or semicolons in the PDDL.
•  If a precondition or effect is empty, write `()` — never `(and)`.
•  The `:requirements` list must exactly match the features you use
   (typically just `:strips :typing`; add `:negative-preconditions`
   *only* if you actually use `not`).
•  Declare at least one object for every type you introduce.
"""

USER_PROMPT_TEMPLATE = """
Environment: {env_name}   (level “{level_name}”)
Category   : {category_name}
Skill      : {skill}

Level description:
{level_description}

Current Agent python code:
{agent_code}

{prev_pddls}

Write DOMAIN and PROBLEM so that a plan exists using *only* the high-level
actions above (plus any brand-new actions you define following the
guidelines).  You are encouraged to invent whatever additional actions are
useful, as long as they obey the naming & abstraction rules.
Remember:
* Declare :types and at least one object per type.
* No comments, no `(and)` empty blocks.
* Avoid `not` (or add :negative-preconditions if you really need it).
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
        self.client = ChatGPTClient("openai/o3", PDDLResp)
        # self.client = ChatGPTClient("ollama/deepseek-r1:8b", PDDLResp)
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

        prev_pddls = ""
        if TMP_DOMAIN.exists() and TMP_PROBLEM.exists():
            prev_pddls = f"Previous PDDL files:\nDomain:\n{TMP_DOMAIN.read_text()}\nProblem:\n{TMP_PROBLEM.read_text()}\n"
        # else:
        #     prev_pddls = "Note: atleast 1 action must precisely match the Category name! \n"

        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(
                env_name=meta["env_name"],
                level_name=meta["level_name"],
                category_name=meta["category_name"],
                skill=meta["skill"],
                level_description=meta["level_description"],
                agent_code=agent_src,
                prev_pddls=prev_pddls
            )}
        ]

        # retry-repair loop
        for attempt in range(1, MAX_RETRIES + 1):
            print(conversation[-1])
            resp: PDDLResp = self.client.chat_completion(conversation)
            print("\n", resp.model_dump_json(indent=2))
            dom_txt, prob_txt = resp.domain.strip(), resp.problem.strip()
            TMP_DOMAIN.write_text(dom_txt)
            TMP_PROBLEM.write_text(prob_txt)
            # dom_txt, prob_txt = TMP_DOMAIN.read_text().strip(), TMP_PROBLEM.read_text().strip()

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
        # try both snake_case and kebab-case
        return (self._schemas.get(act_name)
                or self._schemas.get(act_name.replace("_", "-"), ""))

    def get_action_schemas(self, act_names: Set[str]) -> Dict[str, str]:
        return {a: self.get_action_schema(a) for a in act_names}

