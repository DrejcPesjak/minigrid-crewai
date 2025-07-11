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
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

import agent_tmp
from llmclient import ChatGPTClient
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
#  CONSTANTS
# --------------------------------------------------------------------------- #

AGENT_FILE = Path(__file__).with_name("agent.py")
TMP_FILE   = Path(__file__).with_name("agent_tmp.py")

MAX_ROUNDS = 5

CODER_SYSTEM_PROMPT = """
You are updating the Python `Agent` class for a MiniGrid agent.

Hard rules
----------
• Return **raw Python only** – no markdown fences.  
• Start every `def` at column-0; no wrapper classes.  
• Implement *all* requested high-level actions plus any helper predicates.  
• Actions must return a `list[int]` or be a generator yielding ints.  
• Never mutate `full_grid` or other state vars; read-only.  
• Add required `import …` lines at column-0 if a new std-lib symbol is used.  
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
""".strip()

CODER_FEEDBACK_TEMPLATE = """
Previous patch failed.

--- ERROR / TRACE ----------------------------------
{error_log}

Please output a full replacement for the previously sent code block.
Only raw Python.
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

    def __init__(self):
        # self.client = ChatGPTClient("gemini/gemini-2.5-flash", CodeResp)
        self.client = ChatGPTClient("openai/codex-mini-latest", CodeResp)

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
            print(f"CoderLLM: round {rnd}, prompt tokens~{2*sum(len(m['content'].split()) for m in conversation)}")
            patch: CodeResp = self.client.chat_completion(conversation)

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
    