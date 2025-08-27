from pydantic import BaseModel
from pathlib import Path
import shutil, ast, textwrap, traceback, time
from typing import Dict, Optional, Set
from dataclasses import dataclass

from spca_llm_ur5.llm.llmclient import ChatGPTClient
# from spca_llm_ur5.nodes.ctx_runtime import Ctx
from spca_llm_ur5.scripts.runtime_paths import BASE_ACTS, TMP_ACTS, CTX_PATHS

ACTIONS_FILE     = BASE_ACTS
ACTIONS_TMP_FILE = TMP_ACTS
CTX_FILE         = CTX_PATHS

MAX_ROUNDS = 5
CODER_SYSTEM_PROMPT = """
You are augmenting the Python **module** `agent_actions.py` that controls a **UR5 + Robotiq 2F85** robot 
operating in a **Gazebo + ROS2** world via **MoveItPy**. The robot arm is mounted on a workbench. 
A **Kinect camera** is positioned above the workbench, providing a top-down view. In the camera's feed, 
the robot arm appears at the bottom-center, while the objects to be manipulated are located on the table above it. 
There is **no class**; you will be adding or replacing **top-level functions** only.

HARD RULES 
- **Return only raw Python source** — never wrap in markdown.
- **Top-level `def` blocks only** — no classes; every function must start at column-0. No extra indentation.
- **Actions vs. Helpers**:
  - **Actions** are the public functions called by the plan. Their signature **must be** `def name(ctx, *string_args, done_callback=None)`. The number of `string_args` must precisely match the PDDL.
  - **Actions must NOT return a value**. They must call `done_callback(success=True)` at the end of their execution.
  - **Helpers** (names start with `_`) can have any signature and may return values. They do not need `done_callback`.
- **Use only the provided runtime context `ctx`**. Never create or modify ROS entities directly. All ROS communication must go through the provided `ctx` instance.
- **No shell/subprocess calls**, no file I/O, no sleeps.
- **Plans pass only symbolic strings** (e.g., `"box_blue"`, `"plate_red"`). Do not use numeric coordinates in action signatures.
- **Error handling**: If a precondition fails (e.g., no depth/rgb/TF/plan), `raise RuntimeError("reason")` to abort.

GUIDELINES & PATTERNS
- **Motion**:
  - All arm movements must follow a two-step procedure:
    1.  **Approach**: First, plan a safe, obstacle-avoiding trajectory to a position 20 cm above or away from the goal using `_move_via_ik`.
    2.  **Execute**: Then, use `_cartesian_runner_async` to perform a precise, straight-line movement from the approach position to the final goal.
  - **Never use** `_move_to_pose_stamped` as it is non-functional.
  - Use `ctx.arm` or `ctx.gripper` for MoveItPy planning.
- **`done_callback`**:
  - Actions and asynchronous helper chains **must** call `done_callback(success=True)` upon successful completion or `done_callback(success=False, msg="reason")` upon failure.
  - Synchronous helper functions should not have a `done_callback` parameter.
- **Perception**:
  - Use `ctx.latest_rgb` (BGR8), `ctx.latest_depth` (meters), and `ctx.K_rgb`/`K_depth` for camera data.
- **Joints**:
  - Normalize all joint angles to `[-π, π]` using modulo-2π arithmetic.
  - The `shoulder_lift_joint` angle must be within `[-π, 0]` radians.
- **Cancellation**:
  - For any long-running Python logic (e.g., a `while` loop), check `if ctx.cancelled(): raise RuntimeError("cancelled")`.

Output only the added or modified function `def` blocks (plus any required `import` statements). No markdown, no comments outside the code.
""".strip()


CODER_INITIAL_TEMPLATE = """
Current actions module (agent_actions.py / tmp merge base):
```python
{agent_src}
```

The whole `ctx` class definition is here for your reference. All of its attributes, services, and methods are available via the `ctx` object passed to every function.
```python
{ctx_src}
```

Current state snapshot (for context only):
{agent_state}

Implement **all** of the following PDDL actions as **top-level functions**:
{schemas_text}

The high-level plan to satisfy is:
{plan_str}

Output only the added or modified function `def` blocks (plus any required `import` statements).
No markdown, no extra text.
""".strip()


CODER_FEEDBACK_TEMPLATE = """
Previous patch failed for the UR5 + 2F85 ROS-2 world.

--- ERROR / TRACE ----------------------------------
{error_log}

Fix the issues and output a **complete replacement** for your previous code block.

Return only raw Python source for the replacement functions (and any needed imports). No markdown.
""".strip()


class CodeResp(BaseModel):
    code: str


@dataclass
class CoderResult:
    status: str  # ok | merge_error | reload_error | exhausted
    trace: str = ""


class CoderLLM:
    """
    One call implements ALL `missing_actions`.
    Internal chat-repair loop stops as soon as agent_tmp reloads cleanly.
    """

    def __init__(self):
        # self.client = ChatGPTClient("gemini/gemini-2.5-flash", CodeResp)
        self.client = ChatGPTClient("openai/codex-mini-latest", CodeResp)

    def implement_actions(
        self,
        actions: Set[str],
        pddl_schemas: Dict[str, str],
        plan_str: str,
        agent_state: dict,
        past_error_log: Optional[str] = None,
    ) -> CoderResult:

        # --- choose base source ---
        if not ACTIONS_TMP_FILE.exists():
            shutil.copy2(ACTIONS_FILE, ACTIONS_TMP_FILE)
        
        base_src = ACTIONS_TMP_FILE.read_text()#encoding="utf-8")
        ctx_src = CTX_FILE.read_text()#encoding="utf-8")

        # --- build conversation ---
        schemas_txt = "\n\n".join(pddl_schemas.get(a, "") for a in actions)
        conversation = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT},
            {"role": "user", "content": CODER_INITIAL_TEMPLATE.format(
                agent_src=base_src,
                ctx_src=ctx_src,
                agent_state=agent_state,
                schemas_text=schemas_txt,
                plan_str=plan_str,
            )}
        ]
        if past_error_log:
            conversation.append({
                "role": "user",
                "content": CODER_FEEDBACK_TEMPLATE.format(error_log=past_error_log)
            })

        current_src = base_src

        for rnd in range(1, MAX_ROUNDS + 1):
            print(f"CoderLLM: round {rnd}")
            print(f"CoderLLM tokens approx: {round(sum(len(m['content']) for m in conversation)/3.75)}")
            try:
                patch: CodeResp = self.client.chat_completion(conversation)
                code_block = patch.code.strip()
            except Exception as exc:
                return CoderResult("reload_error", trace=f"LLM call failed: {exc}")

            merged = self._merge_module(current_src, code_block)
            if merged.startswith("Error"):
                err = merged
                status = "merge_error"
            else:
                ok, why = self._syntax_ok(merged)  # AST parse only
                if ok:
                    ACTIONS_TMP_FILE.write_text(merged, encoding="utf-8")
                    return CoderResult("ok")
                status, err = "reload_error", why

            # feedback & retry
            conversation.extend([
                {"role": "assistant", "content": code_block},
                {"role": "user", "content": CODER_FEEDBACK_TEMPLATE.format(error_log=err)}
            ])
            # keep current_src as-is to avoid accumulating bad merges

        return CoderResult("exhausted", trace="exceeded MAX_ROUNDS")

    def _merge_module(self, original: str, patch: str) -> str:
        """Merge top-level imports and function defs by name; no classes."""
        try:
            p_ast = ast.parse(textwrap.dedent(patch))
            b_ast = ast.parse(original)

            p_funcs = {n.name: n for n in p_ast.body if isinstance(n, ast.FunctionDef)}
            p_imps = [n for n in p_ast.body if isinstance(n, (ast.Import, ast.ImportFrom))]

            # split base
            b_imps = [n for n in b_ast.body if isinstance(n, (ast.Import, ast.ImportFrom))]
            b_rest = [n for n in b_ast.body if not isinstance(n, (ast.Import, ast.ImportFrom))]

            # replace or append functions
            new_rest = []
            for n in b_rest:
                if isinstance(n, ast.FunctionDef) and n.name in p_funcs:
                    new_rest.append(p_funcs.pop(n.name))  # overwrite
                else:
                    new_rest.append(n)
            new_rest.extend(p_funcs.values())  # brand-new funcs

            # merge imports (dedupe by AST dump)
            def k(n): return ast.dump(n, annotate_fields=False)
            seen = {k(n) for n in b_imps}
            imps = list(b_imps)
            for n in p_imps:
                kn = k(n)
                if kn not in seen:
                    imps.append(n)
                    seen.add(kn)

            b_ast.body = imps + new_rest
            return ast.unparse(b_ast)
        except Exception as exc:
            return f"Error merging patch: {exc}"

    def _syntax_ok(self, src: str) -> tuple[bool, str]:
        try:
            ast.parse(src)
            return True, ""
        except Exception as exc:
            return False, f"{exc}\n{traceback.format_exc()}"
