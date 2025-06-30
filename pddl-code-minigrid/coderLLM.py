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

import importlib, agent_tmp

from llmclient import ChatGPTClient
from pydantic import BaseModel

AGENT_FILE = Path(__file__).with_name("agent.py")
TMP_FILE   = Path(__file__).with_name("agent_tmp.py")
MAX_ROUNDS = 10

CODER_SYSTEM_PROMPT = """
You are augmenting the Python `Agent` class that controls an OpenAI
Gymnasium MiniGrid agent.

Hard rules
• **Return only raw Python source** - never wrap in markdown.
• Output plain Python.  Start every `def` at column-0 (no extra indent,
  no class wrapper).  The merge script will insert each def into the
  Agent class automatically.
• Never run shell commands, subprocess calls, or print diagnostics.
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

Output only the added or modified `def` blocks.
Each function must start at column-0 (no leading spaces, no class wrapper).
""".strip()

CODER_FEEDBACK_TEMPLATE = """
The plan failed to execute.

--- ERROR / TRACE ---
{error_log}

Produce a *complete replacement* for the previously returned code block.
Start every `def` at column-0 (no leading spaces).
Fix every issue revealed by the error.
""".strip() 


class CodeResp(BaseModel):
    code: str


class CoderLLM:
    def __init__(self):
        # gpt-4.1-mini, gpt-4.1-nano, gpt-4o-mini, o1-mini, o3-mini, o4-mini, and codex-mini-latest
        # self.client = ChatGPTClient("openai/o1", CodeResp)
        self.client = ChatGPTClient("openai/codex-mini-latest", CodeResp)
        # self.client = ChatGPTClient("ollama/deepseek-r1:8b", CodeResp)

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
            # print(conversation[-1])
            print(f"CoderLLM tokens approx: {2 * sum(len(m['content'].split()) for m in conversation)}")
            patch: CodeResp = self.client.chat_completion(conversation)
            print("\n", patch)
            new_src = self._merge(current_src, patch.code.strip())

            if new_src.startswith("Error"):
                outcome = new_src
            else:
                TMP_FILE.write_text(new_src)

                try:
                    importlib.reload(agent_tmp) # ie. import error 
                except Exception as exc:
                    outcome = f"Error reloading agent_tmp: {exc}"
                else:

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
            # # smaller token usage: only last patch + feedback
            # # (but this loses the context of previous patches)
            # conversation = [
            #     conversation[0],                       # system
            #     {"role": "assistant", "content": patch.code},          # last patch only
            #     {"role": "user",      "content": CODER_FEEDBACK_TEMPLATE.format(
            #         error_log=outcome,
            #         schemas_text=schemas_text,
            #         plan_str=plan_str
            #     )}
            # ]
            # current_src = new_src
        
        print(f"❌ CoderLLM failed to implement actions after {round_} rounds")
        raise RuntimeError("CoderLLM exhausted retries") # should i raise here?

    # ------------------------------------------------ helpers ------------
    def _merge(self, original: str, patch: str) -> str:
        """Insert / replace defs INSIDE class Agent."""
        try:
            patch = textwrap.dedent(patch)           # <-- new
            o_ast  = ast.parse(original)
            p_ast  = ast.parse(patch)

            # collect new defs
            repl = {n.name: n for n in p_ast.body if isinstance(n, ast.FunctionDef)}

            # find the Agent class node
            agent_cls = next(n for n in o_ast.body if isinstance(n, ast.ClassDef) and n.name == "Agent")

            # build a new body for the class
            new_body = []
            for node in agent_cls.body:
                if isinstance(node, ast.FunctionDef) and node.name in repl:
                    new_body.append(repl.pop(node.name))
                else:
                    new_body.append(node)
            new_body.extend(repl.values())
            agent_cls.body = new_body

            return ast.unparse(o_ast)
        
        except Exception as exc:
            return f"Error merging patch: {exc}"

    def _test_full_plan(self, plan_str: str, env_name: str) -> str:
        from minigridenv import MiniGridEnv
        env = MiniGridEnv(env_name)
        print(f"Plan to execute in {env_name}:\n{plan_str}")
        result = env.run_sim(plan_str)
        print(f"Plan execution result: {result}")
        env.end_env()
        return result
