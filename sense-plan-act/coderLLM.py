"""
CoderLLM – adds or patches missing Agent actions.

Differences from the old version
• Does *not* run MiniGridEnv – main() owns execution / checkpoint logic.
• Returns a structured CoderResult so main can decide what to do next.
• Prompt now includes the current `agent_state` snapshot.
"""

import ast
import importlib
import shutil
import time
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

import agent_tmp
from llmclient import ChatGPTClient
from pydantic import BaseModel
from metrics_logger import MetricsLogger

# --------------------------------------------------------------------------- #
#  CONSTANTS
# --------------------------------------------------------------------------- #

AGENT_FILE = Path(__file__).with_name("agent.py")
TMP_FILE   = Path(__file__).with_name("agent_tmp.py")

MAX_ROUNDS = 5

CODER_SYSTEM_PROMPT = """
You are augmenting the Python `Agent` class that controls an OpenAI
Gymnasium MiniGrid agent.

Hard rules
• **Return only raw Python source** - never wrap in markdown.
• Output plain Python.  Start every `def` at column-0 (no extra indent,
  no class wrapper).  The merge script will insert each def into the
  Agent class automatically.
• Never run shell commands, subprocess calls, or print diagnostics. 
• To communicate issues raise a RuntimeError with a descriptive message in code.
• Implement every high-level action given, plus any helper predicates the
  PDDL preconditions/effects require.
• Each generated action **must include every parameter that appears in the
  PDDL schema** (keep the same order). If a parameter isn't used inside
  the body, keep it anyway (you can prefix its name with “_” to silence
  linters).
• Re-use existing helpers when possible (am_next_to, lava_ahead, …).
• All **actions** must return either a `list[int]` or be a generator
  (`yield` / `yield from`) producing primitive codes one by one.
• Use `yield from` whenever the code needs to re-inspect `full_grid`
  between moves (e.g. chase, explore, corridor following).
• Never mutate `full_grid`, `current_observation`, `current_dir`,
  `agent_pos`, or `prev_underlying`. Read-only only.
• Predicates return `bool`.
• If you reference a new symbol from any library (e.g. deque, heapq, 
  Callable), add the corresponding `import …` at column-0.

Guidelines
• **Perception model** - `current_observation` is the agent's 7 x 7 egocentric
  view at the *current* step; `full_grid` is an ever-growing global map that
  is padded/updated after every primitive move, and many objects or targets 
  won't be visible at the start.  Plan path-finding or loop conditions against 
  `full_grid`, but be ready to re-query it between moves.
• Prefer `np.where(...)` over hard-coded offsets.
• Avoid infinite loops: the runner aborts the **entire program** if no
  cell change is detected for 5 consecutive steps.
• For multi-step actions that **don't** need fresh perception, just
  `return [2, 2, 1, 2]`.
• Always sanitise PDDL strings: convert kebab-case → snake_case and drop
  colour suffixes when matching object names.
• Keep helper predicates small and reusable (`is_door`, `is_goal`, …).
• **Grid vocabulary** - every cell string is a space-separated combo of  
  OBJECT ∈ {unseen, empty, wall, floor, door, key, ball, box, goal, lava, agent}  
  + optional COLOR ∈ {red, green, blue, purple, yellow, grey}  
  + optional STATE ∈ {open, closed, locked}.  
  No other words ever appear, and the order is always “object [color] [state]”.
• When navigating, note that agent cannot move through objects, it must move around them.

""".strip()

CODER_INITIAL_TEMPLATE = """
Current Agent code:
{agent_src}

Current agent_state snapshot (for context only):
{agent_state}

Implement **all** of the following PDDL actions:

{schemas_text}

The plan that must succeed is:
{plan_str}

Output only the added or modified `def` blocks.
Each function must start at column-0 (no leading spaces, no class wrapper).
Never search the repo — assume the method is absent and implement it.
""".strip()

CODER_FEEDBACK_TEMPLATE = """
Previous patch failed.

--- ERROR / TRACE ----------------------------------
{error_log}

Produce a *complete replacement* for the previously returned code block.
Start every `def` at column-0 (no leading spaces).
Fix every issue revealed by the error.
""".strip()

# --------------------------------------------------------------------------- #
#  Pydantic / dataclasses
# --------------------------------------------------------------------------- #

class CodeResp(BaseModel):
    code: str

@dataclass
class CoderResult:
    status : str              # ok | merge_error | reload_error | exhausted
    trace  : str = ""

# --------------------------------------------------------------------------- #
#  CoderLLM
# --------------------------------------------------------------------------- #

class CoderLLM:
    """
    One call implements ALL `missing_actions`.
    Internal chat-repair loop stops as soon as agent_tmp reloads cleanly.
    """

    def __init__(self, metrics: Optional[MetricsLogger] = None):
        # self.client = ChatGPTClient("gemini/gemini-2.5-flash", CodeResp)
        self.client = ChatGPTClient("openai/codex-mini-latest", CodeResp)
        self.metrics = metrics     # optional MetricsLogger

    # ------------------------------------------------------------------ #

    def implement_actions(
        self,
        actions: Set[str],
        pddl_schemas: Dict[str, str],
        plan_str: str,
        agent_state: dict,
        past_error_log: Optional[str] = None,
    ) -> CoderResult:
        """
        Returns CoderResult(status, trace).
        """

        if not TMP_FILE.exists():
            shutil.copy(AGENT_FILE, TMP_FILE)

        agent_src   = TMP_FILE.read_text()
        schemas_txt = "\n\n".join(pddl_schemas[a] for a in actions)

        conversation = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT},
            {"role": "user",   "content": CODER_INITIAL_TEMPLATE.format(
                agent_src   = agent_src,
                agent_state = agent_state,
                schemas_text= schemas_txt,
                plan_str    = plan_str,
            )}
        ]

        if past_error_log:
            conversation.append({
                "role": "user",
                "content": CODER_FEEDBACK_TEMPLATE.format(error_log=past_error_log)
            })

        current_src = agent_src

        for rnd in range(1, MAX_ROUNDS + 1):
            # print(conversation[-1])  # debug: print last user prompt
            print(f"CoderLLM: syntax round {rnd}, prompt tokens~{2*sum(len(m['content'].split()) for m in conversation)}")

            t0 = time.time()
            patch: CodeResp = self.client.chat_completion(conversation)
            duration = time.time() - t0
            if self.metrics:
                self.metrics.log_coder_call_duration(duration_s=duration)

            merged = self._merge(current_src, patch.code.strip())
            if merged.startswith("Error"):
                err = merged
                status = "merge_error"
            else:
                TMP_FILE.write_text(merged)
                try:
                    importlib.reload(agent_tmp)
                    return CoderResult("ok")
                except Exception as exc:
                    status = "reload_error"
                    err    = f"{exc}\n{traceback.format_exc()}"

            # --- feedback & retry loop -----------------
            conversation.extend([
                {"role": "assistant", "content": patch.code},
                {"role": "user",      "content": CODER_FEEDBACK_TEMPLATE.format(
                    error_log=err)}
            ])
            if self.metrics:
                self.metrics.log_coder_syntax_repair(repair_step=rnd, error_msg=err)
            # current_src = merged       # even if bad, keep context

        return CoderResult("exhausted", trace="exceeded MAX_ROUNDS")


    # ---------------- AST merge ----------------------------------------

    def _merge(self, original: str, patch: str) -> str:
      try:
          patch_ast = ast.parse(textwrap.dedent(patch))
          base_ast  = ast.parse(original)

          # ── split patch into defs vs. imports ────────────────────────────
          new_funcs   = {n.name: n for n in patch_ast.body
                        if isinstance(n, ast.FunctionDef)}
          patch_imps  = [n for n in patch_ast.body
                        if isinstance(n, (ast.Import, ast.ImportFrom))]

          # ── update Agent class body ──────────────────────────────────────
          agent_cls = next(n for n in base_ast.body
                          if isinstance(n, ast.ClassDef) and n.name == "Agent")

          updated_body = []
          for node in agent_cls.body:
              if isinstance(node, ast.FunctionDef) and node.name in new_funcs:
                  updated_body.append(new_funcs.pop(node.name))   # overwrite
              else:
                  updated_body.append(node)
          updated_body.extend(new_funcs.values())                 # append brand-new
          agent_cls.body = updated_body

          # ── merge imports without duplicates ────────────────────────────
          def _key(n): return ast.dump(n, annotate_fields=False)
          existing_imps = [n for n in base_ast.body
                          if isinstance(n, (ast.Import, ast.ImportFrom))]
          exists = {_key(n) for n in existing_imps}

          for imp in patch_imps:
              if _key(imp) not in exists:
                  existing_imps.append(imp)
                  exists.add(_key(imp))

          # keep original non-import top-level nodes
          rest = [n for n in base_ast.body
                  if not isinstance(n, (ast.Import, ast.ImportFrom))]

          base_ast.body = existing_imps + rest
          return ast.unparse(base_ast)

      except Exception as exc:
          tb_str = traceback.format_exc()
          print(tb_str)
          return f"Error merging patch: {exc}"
    