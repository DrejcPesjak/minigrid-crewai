"""
Batch-implement ALL missing actions at once.

Loop:
    • Prompt LLM with current Agent code, every PDDL action schema,
      and the *entire* plan string.
    • Merge patch → agent_tmp.py
    • Run full plan in MiniGrid.
    • On success: copy to agent.py and return.
    • On failure: append error trace + last grid to the conversation and retry.
"""
import ast
import shutil
import textwrap
from pathlib import Path
from typing import Dict, Set

from llmclient import ChatGPTClient
from pydantic import BaseModel
from minigridenv import MiniGridEnv

AGENT_FILE = Path(__file__).with_name("agent.py")
TMP_FILE   = Path(__file__).with_name("agent_tmp.py")
MAX_ROUNDS = 8

CODER_SYSTEM_PROMPT = """
You are augmenting the Python `Agent` class that controls an OpenAI
Gymnasium MiniGrid agent.

Hard rules
• **Return only raw Python source** - never wrap in markdown.
• New code must live *inside* the existing `Agent` class, so indent every
  new line one extra level (8 spaces or 1 tab relative to top level).
• Implement every high-level action given, plus any helper predicates the
  PDDL preconditions/effects require.
• Re-use existing helpers when possible (am_next_to, lava_ahead, …).
• Keep method names identical to the PDDL `:action` names, converting
  kebab-case → snake_case.
• Each action must return `List[int]` of primitive codes the environment
  expects.
• Do **not** modify unrelated parts of the file unless a feedback turn
  explicitly calls out a bug there.
"""

CODER_INITIAL_TEMPLATE = """
Current Agent code:
```python
{agent_src}
````

Implement **all** of the following PDDL actions:

```pddl
{schemas_text}
```

The plan that must succeed is:
{plan_str}

Output only the added or modified `def` blocks (actions + any new
predicates), each correctly indented for class scope.
""".strip()

CODER_FEEDBACK_TEMPLATE = """
The plan failed to execute.

--- ERROR / TRACE ---
{error_log}

Produce a *complete replacement* for the previously returned code block,
adhering to the same formatting rules (raw Python, class-level
indentation).  Fix every issue revealed by the error.
""".strip()


class CodeResp(BaseModel):
    code: str


class CoderLLM:
    def __init__(self):
        self.client = ChatGPTClient("o1", CodeResp)

    # -------------------------------------------------- public ------------
    def implement_actions(
        self,
        actions: Set[str],
        pddl_schemas: Dict[str, str],
        plan_str: str,
        test_env: str,
    ):
        """
        Implement *all* `actions` inside Agent.

        `pddl_schemas` maps action_name → full (:action …) block.
        """
        if not TMP_FILE.exists():
            shutil.copy(AGENT_FILE, TMP_FILE)

        agent_src = TMP_FILE.read_text()
        schemas_text = "\n\n".join(pddl_schemas[a] for a in actions)

        # conversation = [
        #     {"role": "system",
        #      "content": (
        #          "You extend the MiniGrid Agent class. "
        #          "Return ONLY python code (no markdown) containing NEW "
        #          "method definitions and any helper predicates. "
        #          "Each def must have a concise docstring."
        #      )},
        #     {"role": "user",
        #      "content": (
        #          f"Agent code so far:\n```\n{agent_src}\n```\n\n"
        #          "Implement ALL of the following PDDL actions, keeping the "
        #          "semantics faithful:\n\n"
        #          "```pddl\n" + schemas_text + "\n```\n\n"
        #          f"The full plan you must support is:\n{plan_str}"
        #      )},
        # ]
        conversation = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT.strip()},
            {"role": "user",   "content": CODER_INITIAL_TEMPLATE.format(
                agent_src=agent_src,
                schemas_text=schemas_text,
                plan_str=plan_str
            )}
        ]

        current_src = agent_src
        for round_ in range(1, MAX_ROUNDS + 1):
            patch: CodeResp = self.client.chat_completion(conversation)
            new_src = self._merge(current_src, patch.code.strip())
            TMP_FILE.write_text(new_src)

            outcome = self._test_full_plan(plan_str, test_env)
            if outcome == "success":
                shutil.copy(TMP_FILE, AGENT_FILE)
                print("✓ All actions implemented")
                return

            # feedback & retry
            conversation.extend([
                {"role": "assistant", "content": patch.code},
                {"role": "user",      "content": CODER_FEEDBACK_TEMPLATE.format(
                    error_log=outcome)}
            ])
            current_src = new_src

        raise RuntimeError("CoderLLM exhausted retries")

    # ------------------------------------------------ helpers ------------
    def _merge(self, original: str, patch: str) -> str:
        """Replace or append every FunctionDef found in `patch`."""
        o_ast, p_ast = ast.parse(original), ast.parse(patch)
        repl = {n.name: n for n in p_ast.body if isinstance(n, ast.FunctionDef)}

        new_body = []
        for node in o_ast.body:
            if isinstance(node, ast.FunctionDef) and node.name in repl:
                new_body.append(repl.pop(node.name))
            else:
                new_body.append(node)
        new_body.extend(repl.values())
        o_ast.body = new_body
        return ast.unparse(o_ast)

    def _test_full_plan(self, plan_str: str, env_name: str) -> str:
        env = MiniGridEnv(env_name)
        result = env.run_sim(plan_str)
        env.end_env()
        return result
