"""
PDDL generator with repair loop.  Exposes individual action schemas so
main.py (new SPA loop) can forward them to the coder.
"""

import time
import traceback
from pathlib import Path
from typing import Dict, Tuple, Set, Optional

from pydantic import BaseModel
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner, PlanValidator
from unified_planning.engines import PlanGenerationResultStatus

from llmclient import ChatGPTClient
from metrics_logger import MetricsLogger

# --------------------------------------------------------------------------- #
#  CONSTANTS / PROMPTS
# --------------------------------------------------------------------------- #

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
   pick_up, drop, toggle, done, safe_forward, pick_up_obj. DO NOT USE ANY OF THEM.
•  From the **Environment / Category / Skill / Level description**, decide 
   whether the **currently available high-level actions are sufficient**; 
   reuse them only if they can solve the mission exactly,
   precisely match number of parameters (skip the self param). 
   If none fully fit, invent one or more new *snake_case* actions that do,
   that the coder will later implement.
•  In PROBLEM, match object names precisely, use **snake_case** (e.g. `door_red_locked`).
•  Keep DOMAIN compact - a handful of predicates and actions.
•  All predicate / parameter names must match between DOMAIN and PROBLEM.
•  If several versions of an action exist (e.g. action_name, action_name_v2, action_name_v3), 
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
""".strip()

USER_PROMPT_TEMPLATE_FIRST = """
Environment  : {env_name}   (level “{level_name}”)
Category     : {category_name}
Skill        : {skill}

Level description
-----------------
{level_description}

Current snapshot
----------------
Mission   : {mission};
Direction : {direction};
Inventory : {inventory};
Visible grid:
{visible_grid}

Current Agent code (stripped)
-----------------------------
{agent_code}

Write DOMAIN and PROBLEM so that a plan exists using *only* the high-level
actions above (plus any brand-new actions you define following the
guidelines).  You are encouraged to invent whatever additional actions are
useful, as long as they obey the naming & abstraction rules.
Remember:
* Declare :types and at least one object per type.
* No comments, no `(and)` empty blocks.
* Avoid `not` (or add :negative-preconditions if you really need it).
""".strip()
#* Atleast 1 action name must precisely match current Category name (use the latest version)!

USER_PROMPT_TEMPLATE_PDDL = """
{ctx_header}
Previous DOMAIN:
```
{prev_domain}
```
Previous PROBLEM:
```
{prev_problem}
```
"""

USER_PROMPT_TEMPLATE_REPAIR = """
Previous DOMAIN / PROBLEM caused planning failure.

--- ERROR LOG ---
{error_log}

Below are the previous PDDL files; resend **both** fixed files as ONE JSON
object (keys: domain, problem).  No markdown.
""".strip()

# --------------------------------------------------------------------------- #
#  DATA CLASSES
# --------------------------------------------------------------------------- #

class PDDLResp(BaseModel):
    domain: str
    problem: str

class PlanBundle(BaseModel):
    """Returned by PlannerLLM.plan()."""
    domain          : str
    problem         : str
    plan_str        : str
    action_schemas  : Dict[str, str]        # name → full (:action …) text
    up_result       : object                # unified-planning result (kept for debugging)
    up_validation   : object

# --------------------------------------------------------------------------- #
#  HELPERS
# --------------------------------------------------------------------------- #

def _grid_to_string(grid) -> str:
    return "\n".join(", ".join(row) for row in grid)

def _strip_agent_code(src: str) -> str:
    """Keep only defs (with self) and their docstrings to fit prompt budget."""
    out, keep = [], False
    for ln in src.splitlines():
        if ln.lstrip().startswith("def ") and "self" in ln:
            keep = True
            out.append(ln.strip())
            continue
        if keep and (ln.strip().startswith('"""') or ln.strip().startswith("'''")):
            out.append(ln.strip())
            keep = not ln.strip().endswith(('"""',"'''"))
            continue
        if keep:
            out.append(ln.strip())
    return "\n".join(out)

# --------------------------------------------------------------------------- #
#  PLANNER LLM
# --------------------------------------------------------------------------- #

class PlannerLLM:
    """
    *One* call → possibly many LLM repair rounds internally → PlanBundle.
    """

    def __init__(self, metrics: Optional[MetricsLogger] = None):
        self.big_model   = "openai/o3"
        self.small_model = "openai/o4-mini"
        self.client      = ChatGPTClient(self.big_model, PDDLResp)
        self._schemas    = {}          # cached after a success
        self.metrics     = metrics     # optional MetricsLogger
    
    def _set_model(self, model_name: str):
        if self.client.model_name != model_name.split("/")[-1]:
            self.client = ChatGPTClient(model_name, PDDLResp)

    # ---------------------------------- public API ------------------------

    def plan(
        self,
        snapshot: dict,                       # mission, direction, inventory, visible_grid …
        meta: Dict[str, str],                 # category_name, skill, level_name, level_description, env_name
        pddl_hint: Optional[Tuple[str, str]] = None,  # (domain, problem)
        pddl_trusted: bool = False,           # whether to trust the hint
        plan_failed: bool = False,            # whether the previous plan failed
    ) -> PlanBundle:
        """
        Generate (domain, problem), run UP, repair until solved.
        Returns PlanBundle.
        """

        agent_src = Path(__file__).with_name("agent.py").read_text()
        if len(agent_src.split()) > 12_000:               # prompt safety
            agent_src = _strip_agent_code(agent_src)

           
        # ---------- mode & model selection ------------------------------
        if pddl_hint is None:
            print("PlannerLLM: fresh PDDL")
            mode      = "fresh"
            self._set_model(self.big_model)
            ctx_header = ""
        elif pddl_trusted:
            print("PlannerLLM: reusing PDDL")
            mode      = "reuse"
            self._set_model(self.small_model)
            ctx_header = "DOMAIN / PROBLEM below already solved a level in this category."
        elif plan_failed:
            print("PlannerLLM: repairing PDDL")
            mode      = "replan"
            self._set_model(self.big_model)
            ctx_header = ("DOMAIN / PROBLEM below parse correctly, but the resulting "
                          "planner-generated plan failed in simulation. "
                          "Make drastic changes to the PDDL. Make up new actions if needed.")
        else: # if previous level failed completely
            print("PlannerLLM: go from scratch")
            mode      = "else"
            self._set_model(self.big_model)
            ctx_header = ""
            pddl_hint = None  # reset hint to avoid reusing it

        # ---------- build user prompt ------------------------------------
        
        user_msg = USER_PROMPT_TEMPLATE_FIRST.format(
            env_name         = meta["env_name"],
            level_name       = meta["level_name"],
            category_name    = meta["category_name"],
            skill            = meta["skill"],
            level_description= meta["level_description"],
            mission          = snapshot["mission"],
            direction        = snapshot["direction"],
            inventory        = snapshot.get("inventory"),
            visible_grid     = _grid_to_string(snapshot["visible_grid"]),
            # visible_objects  = ", ".join(snapshot.get("visible_objects", [])),
            agent_code       = agent_src,
        )
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg}
        ]

        # append context if we have a previous PDDL
        if pddl_hint is not None:
            prev_domain, prev_problem = pddl_hint
            conversation.append({"role": "user", "content": USER_PROMPT_TEMPLATE_PDDL.format(
                ctx_header   = ctx_header,
                prev_domain  = prev_domain,
                prev_problem = prev_problem
            )})

        # ---------- repair loop -----------------------------------------
        for attempt in range(1, MAX_RETRIES + 1):
            print(f"PlannerLLM tokens approx: {2*sum(len(m['content'].split()) for m in conversation)}")

            t0 = time.time()
            resp: PDDLResp = self.client.chat_completion(conversation)
            duration = time.time() - t0
            
            dom_txt, prob_txt = resp.domain.strip(), resp.problem.strip()
            TMP_DOMAIN.write_text(dom_txt)
            TMP_PROBLEM.write_text(prob_txt)
            # dom_txt, prob_txt = TMP_DOMAIN.read_text().strip(), TMP_PROBLEM.read_text().strip()

            if self.metrics:
                self.metrics.log_planner_mode(kind=mode, model=getattr(self.client, "model_name", "unknown"), duration_s=duration)

            try:
                up_problem, up_result, up_validation = self._solve_and_validate()
                if up_result.status in (
                        PlanGenerationResultStatus.SOLVED_SATISFICING,
                        PlanGenerationResultStatus.SOLVED_OPTIMALLY):
                    plan_str = self._up_plan_to_minigrid(up_result.plan)
                    self._cache_action_schemas(dom_txt)
                    return PlanBundle(
                        domain         = dom_txt,
                        problem        = prob_txt,
                        plan_str       = plan_str,
                        action_schemas = self._schemas,
                        up_result      = up_result,
                        up_validation  = up_validation
                    )
                err_msg = f"UP returned {up_result.status}"
            except Exception as exc:
                err_msg = str(exc)
                traceback.print_exc()

            # ---- build refinement prompt ----
            conversation.extend([
                {"role": "assistant", "content": resp.model_dump_json()},
                {"role": "user",      "content": USER_PROMPT_TEMPLATE_REPAIR.format(
                    error_log    = err_msg,
                    # # These are already in the "role: assistant" message
                    # prev_domain  = dom_txt,
                    # prev_problem = prob_txt
                )}
            ])
            mode = "syntax"
            self._set_model(self.big_model)
            time.sleep(1)

        raise RuntimeError("PlannerLLM exhausted retries")

    # ---------------------------------- helpers -------------------------

    def _solve_and_validate(self):
        """Run Unified-Planning pipeline on TMP_DOMAIN / TMP_PROBLEM."""
        if not TMP_DOMAIN.exists() or not TMP_PROBLEM.exists():
            raise FileNotFoundError("domain.pddl / problem.pddl missing")
        reader   = PDDLReader()
        problem  = reader.parse_problem(str(TMP_DOMAIN), str(TMP_PROBLEM))
        with OneshotPlanner(problem_kind=problem.kind) as planner:
            up_result = planner.solve(problem)
        with PlanValidator(name="tamer") as validator:
            up_validation = validator.validate(problem, up_result.plan)
        return problem, up_result, up_validation

    def _up_plan_to_minigrid(self, up_plan) -> str:
        """Convert UP plan → `[foo(), bar(obj1)]` string."""
        steps = []
        for step in up_plan.actions:
            name = step.action.name.replace("-", "_")
            if step.actual_parameters:
                params = ", ".join(map(str, step.actual_parameters))
                steps.append(f"{name}({params})")
            else:
                steps.append(f"{name}()")
        return "[" + ", ".join(steps) + "]"

    # ---------- schema cache ----------
    def _cache_action_schemas(self, domain_txt: str):
        self._schemas = {}
        buf = []
        for ln in domain_txt.splitlines():
            if ln.lstrip().startswith("(:action"):
                buf = [ln]
            elif buf:
                buf.append(ln)
                if ln.rstrip().endswith(")"):
                    act_name = buf[0].split()[1]
                    self._schemas[act_name] = "\n".join(buf)
                    buf = []

    def get_action_schema(self, act_name: str) -> str:
        return (self._schemas.get(act_name)
                or self._schemas.get(act_name.replace("_","-"), ""))

    def get_action_schemas(self, act_names: Set[str]) -> Dict[str,str]:
        return {a: self.get_action_schema(a) for a in act_names}
