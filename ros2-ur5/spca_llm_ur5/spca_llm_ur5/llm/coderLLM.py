from pydantic import BaseModel
from pathlib import Path
import shutil, ast, textwrap, traceback, time
from typing import Dict, Optional, Set
from dataclasses import dataclass

from spca_llm_ur5.llm.llmclient import ChatGPTClient
from spca_llm_ur5.nodes.ctx_runtime import Ctx
from spca_llm_ur5.scripts.runtime_paths import BASE_ACTS, TMP_ACTS

ACTIONS_FILE     = BASE_ACTS
ACTIONS_TMP_FILE = TMP_ACTS

MAX_ROUNDS = 5
CODER_SYSTEM_PROMPT = """
You are augmenting the Python **module** `agent_actions.py` that controls a **UR5 + Robotiq 2F85** robot
operating in a **Gazebo + ROS2** world via **MoveItPy**. There is **no class**; you add or replace
**top-level functions** only.

HARD RULES
• **Return only raw Python source** — never wrap in markdown. One or more `def` blocks at column-0.
• **Top-level defs only** — no classes; every function starts at column-0. No extra indentation.
• **Actions vs. Helpers**
  - **Actions** are the public functions the plan calls. Signature **must be** `def name(ctx, *string_args)`.
  - **Actions must NOT return anything**, must NOT print or log, must NOT create ROS nodes/subs/clients/actions.
  - **Helpers** (names start with `_`) may return values and can use NumPy/OpenCV/etc., but still **must not create ROS entities**.
• **Use only the provided runtime context `ctx`**:
  - Motion: `ctx.robot`, `ctx.arm` (UR5 group), `ctx.gripper` (2F85 group).
  - TF: `ctx.tfbuf` (use for transforms), frames: `ctx.WORLD_FRAME`, `ctx.EEF_LINK`.
  - Perception caches: `ctx.latest_rgb`, `ctx.latest_depth` (meters), `ctx.K_rgb`, `ctx.K_depth`, `ctx.latest_cloud`,
    `ctx.rgb_frame`, `ctx.depth_frame`.
  - Services/Actions (already created): `ctx.cart_cli` (/compute_cartesian_path),
    `ctx.follow_traj_ac` (/joint_trajectory_controller/follow_joint_trajectory),
    `ctx.gripper_ac` (/gripper_position_controller/gripper_cmd).
  - Joint states cache: `ctx.joint_state`.
  - Cancellation: check `ctx.cancelled()` in long-running logic and raise `RuntimeError("cancelled")` to abort.
• **Never create or modify ROS entities** in your code (no Node(), create_subscription(), ActionClient(), etc.).
  You must use the ones in `ctx` only.
• **No shell/subprocess**, no file I/O, no sleeps. Compute and act deterministically.
• **Plans pass only symbolic strings** (e.g., `"box_blue"`, `"plate_red"`). Do **not** require numeric coordinates in action
  signatures; compute any necessary numeric goals internally (perception/TF/MoveIt).
• **When you need motion**:
  - Use MoveItPy planning component APIs on `ctx.arm`/`ctx.gripper`.
  - Prefer the existing helper pattern `_plan_and_execute(ctx, planning_component)` when available.
  - For Cartesian segments, call `/compute_cartesian_path` via `ctx.cart_cli`, then execute with `ctx.follow_traj_ac`
    (e.g., through a helper like `_move_cartesian_srv` + `_follow_trajectory`).
  - For named states, use `set_start_state_to_current_state()` and `set_goal_state(configuration_name=...)`.
  - For pose goals, create a `PoseStamped` in `ctx.WORLD_FRAME` (or transform from camera frame via TF) and use
    `set_goal_state(pose_stamped=..., pose_link=ctx.EEF_LINK)`.
• **Perception**:
  - RGB: `ctx.latest_rgb` (BGR8); Depth: `ctx.latest_depth` in **meters**; intrinsics: `ctx.K_rgb` / `ctx.K_depth`.
  - If you must back-project a pixel (u,v) with depth z: `x=(u-cx)*z/fx`, `y=(v-cy)*z/fy`, `z=z` in the camera frame.
  - Transform to world using TF from `ctx.depth_frame`/`ctx.rgb_frame` to `ctx.WORLD_FRAME`.
• **Imports**: If you need an external symbol (e.g., `numpy as np`, `cv2`), add a normal `import` at column-0 in your output.
• **Error handling**: If a precondition cannot be met (no depth/rgb/TF/plan), `raise RuntimeError("reason")`. The caller
  handles failures.
• **Naming & signatures**:
  - Keep action function names **exactly** as requested (snake_case).
  - Keep action parameters **exactly** as requested (all strings). Do not add extra parameters.
  - Helpers must start with `_` and may have any signature/returns.
• **Do not mutate global state** outside your function scope; interact via `ctx` and MoveIt only.
• **Determinism**: Avoid random choices; if ambiguity exists, pick a simple, documented heuristic inside the function.
• **Performance**: Avoid heavy per-pixel Python loops when possible; prefer NumPy / vectorized ops.
• **Output requirement**: Produce only valid Python source consisting of one or more `def` blocks and any needed `import`s.

GUIDELINES & PATTERNS
• Use/extend existing helpers if present (e.g., `_plan_and_execute`, `_move_cartesian_srv`, `_follow_trajectory`,
  `_closest_point_3d`, `_move_arm_to_posestamped`, `_move_arm_into_jointconstraints`). If missing, implement them.
• Typical high-level actions (examples; do not hard-code unless asked): `touch(obj)`, `pick_up(obj)`, `place_on(target)`,
  `gripper_open()`, `gripper_close()`, `move_arm_from_home_to_up()`.
• Long actions should start by checking `if ctx.cancelled(): raise RuntimeError("cancelled")`.

• Ensure the start state is explicitly set before planning.
• Use the `set_start_state_to_current_state()` method to set the start state before planning.
• Normalize all joint angles to the range [-π, π] radians using modulo-2π arithmetic before setting the goal state.
• Explicitly ensure that the shoulder_lift_joint angle is not within the range [0, π] radians. Instead, it should fall within either [-π, 0] radians or [π, 2π] radians.
• Only elbow_joint is really limited to [-π, π] radians.

• Robot/world specifics to assume (static):
  - Robot: UR5 manipulator group name: `"ur5_manipulator"`, gripper group: `"robotiq_gripper"`.
  - End-effector link: `tool0`. World frame: `"world"`.
  - Built-in named states commonly used: `"home"`, `"up"`, `"open"`, `"close"`.
  - Topics/servers/actions are already wired inside `ctx`; do not recreate them.

Your job: implement or replace the requested **actions** (and any underscore-helpers they need) so that the plan can execute
in the UR5 + 2F85 Gazebo ROS-2 setup using only the resources in `ctx`. Remember: **no returns for actions; helpers return OK**.
""".strip()


CODER_INITIAL_TEMPLATE = """
Current actions module (agent_actions.py / tmp merge base):
{agent_src}

Runtime context (fixed; available via `ctx`):
• Frames/Groups: WORLD={world_frame}  EEF_LINK={eef_link}  ARM_GROUP={arm_group}  GRIPPER_GROUP={gripper_group}
• Topics: RGB={rgb_topic}, Depth={depth_topic}, RGB Info={rgb_info_topic}, Depth Info={depth_info_topic}, Cloud={cloud_topic}
• Joint States: {joint_states}
• Actions: Follow Trajectory={follow_traj_action}, Gripper Command={gripper_cmd_action}, Cartesian Service={cartesian_srv}

• Perception caches: latest_rgb (BGR8), latest_depth (meters), K_rgb/K_depth (3×3), latest_cloud (XYZ), rgb_frame, depth_frame
• Motion planning: ctx.robot (MoveItPy), ctx.arm (UR5 planning component), ctx.gripper (2F85 planning component)
• Services/Actions: ctx.cart_cli (/compute_cartesian_path), ctx.follow_traj_ac (FollowJointTrajectory), ctx.gripper_ac (GripperCommand)
• Joint states: ctx.joint_state
• TF: ctx.tfbuf (use for transforms), do not create your own listeners
• Cancellation: ctx.cancelled()

Current state snapshot (for context only):
{agent_state}

Implement **all** of the following PDDL actions as **top-level functions**:
(Each action takes `ctx` followed by **string** parameters only; no numbers in signatures.)
{schemas_text}

The high-level plan to satisfy is:
{plan_str}

Implementation requirements for this robot world:
• Produce **only** raw Python code with one or more `def` blocks (and any needed imports) at column-0.
• **Actions**: `def name(ctx, *string_args)` — **no return**, no print/log, no ROS entity creation; use `ctx` exclusively.
• **Helpers**: names start with `_`, may return values; still must not create ROS entities; can use NumPy/OpenCV.
• Use MoveItPy for planning/execution; prefer `_plan_and_execute(ctx, planning_component)` when available.
• For Cartesian segments, call `/compute_cartesian_path` via `ctx.cart_cli` and execute with `ctx.follow_traj_ac`.
• For pose goals, build `PoseStamped` in `ctx.WORLD_FRAME` (or transform from camera frame) and set `pose_link=ctx.EEF_LINK`.
• If you need 3D from depth, back-project using `K_depth` (or `K_rgb`) and transform with TF to the world frame.
• Check `ctx.cancelled()` early in long actions and raise `RuntimeError("cancelled")` to abort cleanly.
• If a prerequisite is missing (no depth/intrinsics/TF), raise `RuntimeError("reason")` instead of printing.
• Keep function names/parameter lists **exactly** as requested by the plan/schemas.
• When moving the arm to a certain position, it is important to consider the approach angle and the orientation of the end-effector.

Output only the added or modified function `def` blocks (plus any required `import` statements).
No markdown, no comments outside the code, no extra text.
""".strip()


CODER_FEEDBACK_TEMPLATE = """
Previous patch failed for the UR5 + 2F85 ROS-2 world.

--- ERROR / TRACE ----------------------------------
{error_log}

Fix the issues and output a **complete replacement** for your previous code block:
• Start every `def` at column-0; top-level functions only; no classes.
• Actions must keep the exact names and **string-only** parameters as required by the plan/schemas.
• Actions must **not** return values, must not print/log, and must not create ROS entities; use only `ctx`.
• Use MoveItPy via `ctx.arm` / `ctx.gripper` and existing helpers (plan/execute, Cartesian path, TF, depth back-projection).
• Add any missing `import` statements at column-0 (e.g., `import numpy as np`, `import cv2`) if you use them.
• If a prerequisite is absent (e.g., no depth/intrinsics/TF), raise `RuntimeError("reason")` instead of printing.

Return only raw Python source for the replacement functions (and any needed imports). No markdown.
""".strip()


ctxenvs = {
    "world_frame": Ctx.WORLD_FRAME,
    "eef_link": Ctx.EEF_LINK,
    "arm_group": Ctx.ARM_GROUP,
    "gripper_group": Ctx.GRIPPER_GROUP,
    "rgb_topic": Ctx.RGB_TOPIC,
    "depth_topic": Ctx.DEPTH_TOPIC,
    "rgb_info_topic": Ctx.RGB_INFO_TOPIC,
    "depth_info_topic": Ctx.DEPTH_INFO_TOPIC,
    "cloud_topic": Ctx.CLOUD_TOPIC,
    "joint_states": Ctx.JOINT_STATES,
    "follow_traj_action": Ctx.FOLLOW_TRAJ_ACTION,
    "gripper_cmd_action": Ctx.GRIPPER_CMD_ACTION,
    "cartesian_srv": Ctx.CARTESIAN_SRV,
}


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

        # --- build conversation ---
        schemas_txt = "\n\n".join(pddl_schemas.get(a, "") for a in actions)
        conversation = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT},
            {"role": "user", "content": CODER_INITIAL_TEMPLATE.format(
                agent_src=base_src,
                agent_state=agent_state,
                schemas_text=schemas_txt,
                plan_str=plan_str,
                **ctxenvs
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
