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

from spca_llm_ur5.llm.llmclient import ChatGPTClient
from spca_llm_ur5.scripts.runtime_paths import RUN_DIR, BASE_ACTS, TMP_ACTS

# --------------------------------------------------------------------------- #
#  CONSTANTS / PROMPTS
# --------------------------------------------------------------------------- #
TMP_DOMAIN  = RUN_DIR / "domain.pddl"
TMP_PROBLEM = RUN_DIR / "problem.pddl"

ACTIONS_FILE       = BASE_ACTS
ACTIONS_TMP_FILE    = TMP_ACTS

MAX_RETRIES = 8

SYSTEM_PROMPT = """
You are a classical-planning expert for a **table-top manipulation robot** operating in a generic simulation.

Return one JSON object only:

    { "domain": "<full PDDL DOMAIN>",
      "problem": "<full PDDL PROBLEM>" }

No markdown, no plan, no commentary.

Abstraction rules
•  Stay semantic — do not use numeric coordinates, continuous geometry, or metric time in the PDDL.
•  You still need a **minimal type system** suited to the language in the task/scene text; declare at least one object per type used.
•  State is expressed only via concise, boolean predicates derived from the task/scene wording (e.g., grasp status, placement relations, separation relations).
•  From the **Task Title / Description / Scene text**, decide whether the **currently available high-level actions are sufficient**; reuse them only if they exactly fit the plan.
•  If none fully fit, invent one or more new *snake_case* actions that do, with **string parameters only**; the coder will implement them later.
•  In PROBLEM, reference **only object names that appear in the task/scene text**, normalized to snake_case (e.g., `cone_green`, `car_blue`).
•  Keep DOMAIN compact — a handful of predicates and actions tailored to the text.
•  All predicate / parameter names must match exactly between DOMAIN and PROBLEM.

Syntax constraints (very important)
•  **Do NOT use `(not …)`** in preconditions/effects unless you also add `:negative-preconditions` to `:requirements`. Prefer to avoid `not`.
•  Do **not** include comments or semicolons in the PDDL.
•  If a precondition or effect is empty, write `()` — never `(and)`.
•  The `:requirements` list must exactly match the features you use
   (typically just `:strips :typing`; add `:negative-preconditions`
   *only* if you actually use `not`).
•  Declare at least one object for every type you introduce.
""".strip()

USER_PROMPT_TEMPLATE_FIRST = """
Task group    : {task_group}
Task id       : {task_id}

Task title
----------
{task_title}

Task description
----------------
{task_description}

Environment description (full, verbatim from perception)
--------------------------------------------------------
{scene_text}

Write DOMAIN and PROBLEM so that a plan exists using *only* the high-level
actions above (plus any brand-new actions you define following the
guidelines).  You are encouraged to invent whatever additional actions are
useful, as long as they obey the naming & abstraction rules.
Remember:
* Declare :types and at least one object per type.
* No comments, no `(and)` empty blocks.
* Avoid `not` (or add :negative-preconditions if you really need it).
""".strip()
#* Prefer concise snake_case aligned with the language used in the task/scene text; parameters are strings only.

USER_PROMPT_TEMPLATE_PDDL = """
{ctx_header}

Previous DOMAIN:
{prev_domain}

Previous PROBLEM:
{prev_problem}
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

def _strip_actions_code(src: str) -> str:
    """
    Keep only top-level `def` headers and any immediately following docstrings.
    """
    out, keep, doc = [], False, False
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith("def ") and not s.startswith("def _"):
            keep = True
            out.append(s)
            doc = False
            continue
        if keep and (s.startswith('"""') or s.startswith("'''")):
            out.append(s)
            if s.endswith('"""') or s.endswith("'''"):
                keep = False
            else:
                doc = True
            continue
        if doc:
            out.append(s)
            if s.endswith('"""') or s.endswith("'''"):
                keep, doc = False, False
    return "\n".join(out)

def _read_actions_source() -> str:
    if ACTIONS_TMP_FILE.exists():
        return ACTIONS_TMP_FILE.read_text(encoding="utf-8")
    if ACTIONS_FILE.exists():
        return ACTIONS_FILE.read_text(encoding="utf-8")
    return ""  # if missing, leave empty

# --------------------------------------------------------------------------- #
#  PLANNER LLM
# --------------------------------------------------------------------------- #

class PlannerLLM:
    """
    *One* call → possibly many LLM repair rounds internally → PlanBundle.
    """

    def __init__(self):
        self.big_model   = "openai/o3"
        self.small_model = "openai/o4-mini"
        self.client      = ChatGPTClient(self.big_model, PDDLResp)
        self._schemas    = {}          # cached after a success
    
    def _set_model(self, model_name: str):
        if self.client.model_name != model_name.split("/")[-1]:
            self.client = ChatGPTClient(model_name, PDDLResp)

    # ---------------------------------- public API ------------------------

    def plan(
        self,
        snapshot: dict,                       # expects: scene_text (full, verbatim)
        meta: Dict[str, str],                 # task_group, task_id, title, description
        pddl_hint: Optional[Tuple[str, str]] = None,  # (domain, problem)
        pddl_trusted: bool = False,           # whether to trust the hint
        plan_failed: bool = False,            # whether the previous plan failed
    ) -> PlanBundle:
        """
        Generate (domain, problem), run UP, repair until solved.
        Returns PlanBundle.
        """

        # --- read current actions source and strip it for prompt budget ---
        actions_src_full = _read_actions_source()
        actions_outline  = _strip_actions_code(actions_src_full)

        # --- snapshot fields (NO SHORTENING of environment description) ---
        scene_text       = snapshot.get("scene_text", "")
        task_group       = meta.get("task_group", "")
        task_id          = meta.get("task_id", "")
        task_title       = meta.get("title", "")
        task_description = meta.get("description", "")

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
            ctx_header = "DOMAIN / PROBLEM below already solved a level in this task group."
        elif plan_failed:
            print("PlannerLLM: repairing PDDL")
            mode      = "replan"
            self._set_model(self.big_model)
            ctx_header = ("DOMAIN / PROBLEM below parse correctly, but the resulting "
                          "planner-generated plan failed in execution. "
                          "Revise abstractions to better fit the task/scene wording.")
        else:
            print("PlannerLLM: go from scratch")
            mode      = "else"
            self._set_model(self.big_model)
            ctx_header = ""
            pddl_hint = None  # reset hint to avoid reusing it

        # ---------- build user prompt ------------------------------------
        user_msg = USER_PROMPT_TEMPLATE_FIRST.format(
            task_group       = task_group,
            task_id          = task_id,
            task_title       = task_title,
            task_description = task_description,
            scene_text       = scene_text,
            actions_outline  = actions_outline,
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
            # print(f"PlannerLLM tokens approx: {2*sum(len(m['content'].split()) for m in conversation)}")
            # resp: PDDLResp = self.client.chat_completion(conversation)
            # dom_txt, prob_txt = resp.domain.strip(), resp.problem.strip()
            # TMP_DOMAIN.write_text(dom_txt)
            # TMP_PROBLEM.write_text(prob_txt)
            dom_txt, prob_txt = TMP_DOMAIN.read_text(encoding="utf-8"), TMP_PROBLEM.read_text(encoding="utf-8")

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

            # # ---- build refinement prompt ----
            # conversation.extend([
            #     {"role": "assistant", "content": resp.model_dump_json()},
            #     {"role": "user",      "content": USER_PROMPT_TEMPLATE_REPAIR.format(
            #         error_log    = err_msg,
            #     )}
            # ])
            mode = "syntax repair"
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
