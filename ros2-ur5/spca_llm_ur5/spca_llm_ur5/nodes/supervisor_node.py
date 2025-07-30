#!/usr/bin/env python3
import os
import re
import json
import shutil
import rclpy
import yaml
from pathlib import Path
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Empty

from spca_llm_ur5.llm.plannerLLM import PlannerLLM
from spca_llm_ur5.llm.coderLLM   import CoderLLM
from spca_llm_ur5.llm.senseLLM   import SenseLLM

import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


from spca_llm_ur5.runtime_paths import BASE_ACTS, TMP_ACTS
ACTIONS_FILE     = BASE_ACTS
ACTIONS_TMP_FILE = TMP_ACTS

SPA_ROUNDS = 5
MAX_CODER_RETRIES = 5

class Supervisor(Node):
    def __init__(self):
        super().__init__('supervisor')
        self.logger = self.get_logger()
        self.declare_parameter('curriculum_yaml', '')

        curr_path = self.get_parameter('curriculum_yaml').get_parameter_value().string_value
        self.levels = self._load_curriculum(curr_path)   # [{group, path, doc}]
        if not self.levels:
            self.logger.error("No levels found in curriculum.")
        # PDDL cache keyed by task_group
        self.group_cache = {}  # group -> {"pddl": (dom, prob) | None, "trusted": bool}

        # pubs/subs
        self.plan_pub        = self.create_publisher(String, '/planner/plan', 10)
        self.code_pub        = self.create_publisher(String, '/coder/script', 10)       # reload trigger only
        self.ref_lvl_pub     = self.create_publisher(String, '/referee/set_level', 10)
        self.exec_cancel_pub = self.create_publisher(String, '/executor/cancel', 10)

        self.task_status = 'running'   # from referee
        self.exec_outcome = None       # from executor
        self.create_subscription(String, '/task/status', self._task_cb, 10)
        self.create_subscription(String, '/executor/outcome', self._exec_cb, 10)

        # world reset
        self.reset_cli = self.create_client(Empty, '/reset_world')
        # self.reset_cli = self.create_client(Empty, '/reset_simulation')
        self.reset_world_in_progress = False

        # loop state
        self.idx = 0
        self.stage = 'idle'
        self.retry_semantic = 0
        self.spa_round = 1

        self.planner = PlannerLLM()
        self.coder   = CoderLLM()
        self.sense   = SenseLLM()

        self._bridge = CvBridge()
        self.last_image_bgr = None
        self.create_subscription(
            Image,
            "/camera/image_raw",
            self._image_cb,
            qos_profile=10) 
        
        self.timer = self.create_timer(0.1, self._tick)

        self.logger.info(f"Supervisor ready; {len(self.levels)} level(s) loaded.")

    # -------- helpers ----------
    def _task_cb(self, msg: String):
        # from referee: running | success | fail | timeout
        self.task_status = msg.data

    def _exec_cb(self, msg: String):
        # from executor: JSON string with status/msg/trace
        try:
            self.exec_outcome = json.loads(msg.data)
        except Exception:
            self.exec_outcome = {"status": "parse_error", "msg": msg.data, "trace": ""}

    # def _reset_world(self):
    #     if not self.reset_cli.service_is_ready():
    #         self.reset_cli.wait_for_service(timeout_sec=5.0)
    #     fut = self.reset_cli.call_async(Empty.Request())
    #     rclpy.spin_until_future_complete(self, fut)

    def _reset_world_async(self):
        if self.reset_world_in_progress:
            self.logger.warn("Reset world already in progress.")
            return

        if not self.reset_cli.service_is_ready():
            self.logger.info("Waiting for /reset_world service...")
            # It's generally better to wait for the service once, possibly in __init__
            # or before the first time you might need it, rather than every time.
            # However, for robustness, checking here and logging is fine.
            self.reset_cli.wait_for_service(timeout_sec=5.0)
            if not self.reset_cli.service_is_ready():
                self.logger.error("Reset world service not available after waiting.")
                # You might want to transition to an error state or skip the level here
                self.stage = 'idle' # Revert to idle to try again or move on
                return

        self.logger.info("Calling /reset_world service asynchronously...")
        self.reset_world_in_progress = True
        fut = self.reset_cli.call_async(Empty.Request())
        # Attach a callback to the future
        fut.add_done_callback(self._reset_world_response_cb)
    
    def _reset_world_response_cb(self, future):
        self.reset_world_in_progress = False
        try:
            response = future.result()
            self.logger.info("Reset world service call completed.")
            # Now that reset is done, transition to the next stage
            # This is crucial: the stage change happens *after* the service completes
            self.task_status = 'running' # Reset task status after world reset
            self.exec_outcome = None
            self.retry_semantic = 0
            self.spa_round = 1
            self.stage = 'plan'
        except Exception as e:
            self.logger.error(f"Reset world service call failed: {e}")
            # Handle error: perhaps retry or skip level
            self.stage = 'idle' # Revert to idle to try again

    def _abs_from(self, base_file: str, rel: str) -> str:
        return rel if os.path.isabs(rel) else os.path.join(os.path.dirname(base_file), rel)

    def _load_curriculum(self, curr_yaml_path: str):
        """Return list of {group, path, doc}"""
        doc = yaml.safe_load(open(curr_yaml_path, 'r'))
        out = []
        for g in doc.get('groups', []):
            gname = g.get('name', 'group')
            for p in g.get('levels', []):
                lvl_path = self._abs_from(curr_yaml_path, p)
                try:
                    lvl_doc = yaml.safe_load(open(lvl_path, 'r'))
                except Exception:
                    lvl_doc = {}
                out.append({'group': gname, 'path': lvl_path, 'doc': lvl_doc})
        if not out:
            # fallback: legacy top-level "levels"
            for p in doc.get('levels', []):
                lvl_path = self._abs_from(curr_yaml_path, p)
                try:
                    lvl_doc = yaml.safe_load(open(lvl_path, 'r'))
                except Exception:
                    lvl_doc = {}
                out.append({'group': 'default', 'path': lvl_path, 'doc': lvl_doc})
        return out

    # def _get_img(self, timeout: float = 1.0):
    #     """Fetch a single /camera/image_raw frame. Returns BGR np.uint8 or None."""
    #     box = {"bgr": None}
    #     self.logger.info(f"Waiting for camera image (timeout={timeout}s)...")

    #     def _cb(msg: Image):
    #         try:
    #             box["bgr"] = self._bridge.imgmsg_to_cv2(msg, "bgr8")
    #         except Exception:
    #             box["bgr"] = None

    #     sub = self.create_subscription(Image, "/camera/image_raw", _cb, 10)
    #     try:
    #         end = self.get_clock().now().nanoseconds + int(timeout * 1e9)
    #         while box["bgr"] is None and self.get_clock().now().nanoseconds < end:
    #             rclpy.spin_once(self, timeout_sec=1.05)
    #             self.logger.info("Waiting for image...")
    #     finally:
    #         self.destroy_subscription(sub)

    #     return box["bgr"]

    def _image_cb(self, msg: Image):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if msg.encoding == "rgb8":            # convert once here
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.last_image_bgr = img
        except CvBridgeError:
            self.logger.error("Failed to convert image from CvBridge.")

    # Called from the SPA loop
    def _get_img(self, wait_s: float = 1.0):
        self.logger.info(f"Waiting for camera image (timeout={wait_s}s)...")
        if self.last_image_bgr is not None:
            return self.last_image_bgr          # already have one
        # Otherwise wait up to wait_s for the first frame
        end = self.get_clock().now() + rclpy.time.Duration(seconds=wait_s)
        while self.last_image_bgr is None and self.get_clock().now() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            self.logger.info("Waiting for camera image...")
        return self.last_image_bgr 

    @staticmethod
    def _parse_plan_names(plan_str: str):
        """Return set of action names from a '[foo(), bar(x)]' string."""
        pat = re.compile(r'([A-Za-z][\w-]*)\s*\(')
        return {m.group(1).replace("-", "_") for m in pat.finditer(plan_str or "")}

    @staticmethod
    def _function_names_from_file(py_path: Path):
        """Top-level function names in file (exclude _private)."""
        try:
            import ast
            src = py_path.read_text(encoding="utf-8")
            tree = ast.parse(src)
            names = set()
            for n in tree.body:
                if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
                    names.add(n.name)
            return names
        except Exception:
            return set()

    def _ensure_group_cache(self, group: str):
        if group not in self.group_cache:
            self.group_cache[group] = {"pddl": None, "trusted": False}

    # -------- SPA loop ----------
    def _tick(self):
        self.logger.info(f"Tick: idx={self.idx}, stage={self.stage}, spa_round={self.spa_round}, retry_semantic={self.retry_semantic}")
        if self.idx >= len(self.levels):
            return

        lvl_entry = self.levels[self.idx]
        lvl_path  = lvl_entry['path']
        group     = lvl_entry['group']
        lvl_doc   = lvl_entry['doc'] or {}
        self._ensure_group_cache(group)

        # Extract meta from level doc (minimal contract)
        meta = {
            "task_group": group,
            "task_id": lvl_doc.get("task_id", Path(lvl_path).stem),
            "title": lvl_doc.get("title", ""),
            "description": lvl_doc.get("description", ""),
        }

        if self.stage == 'idle':
            self.logger.info(f"IDLE → waiting for next level")
            self.logger.info(f"=== [{group}] {self.idx+1}/{len(self.levels)}: {lvl_path}")
            # reset world and counters
            self._reset_world_async()
            # self._reset_world()
            # self.task_status = 'running'
            # self.exec_outcome = None
            # self.retry_semantic = 0
            # self.spa_round = 1
            # self.stage = 'plan'
            return

        if self.stage == 'plan':
            # —— SENSE (once per SPA round) —— #
            self.logger.info(f"SENSE → {meta['task_id']}")
            img = self._get_img(wait_s=10.0)
            if img is None:
                self.logger.warn("No camera frame; proceeding with empty scene_text.")
            else:
                self.logger.info(f"Got camera image of shape {img.shape}")
            try:
                scene_text = self.sense.describe(
                    title=meta['title'],
                    description=meta['description'],
                    bgr_image=img
                )
                self.logger.info(f"Scene text: {scene_text}")
            except Exception as e:
                self.logger.warn(f"SenseLLM failed: {e}; using empty scene_text.")
                scene_text = ""

            # —— PLAN (LLM/PDDL) —— #
            self.logger.info(f"PLAN → {meta['task_id']}")
            cache = self.group_cache[group]
            try:
                bundle = self.planner.plan(
                    snapshot={"scene_text": scene_text},
                    meta=meta,
                    pddl_hint=cache["pddl"],
                    pddl_trusted=cache["trusted"],
                    plan_failed=(self.retry_semantic >= MAX_CODER_RETRIES)
                )
            except Exception as e:
                self.logger.error(f"PlannerLLM error: {e}")
                # Treat as replan next tick
                self.stage = 'plan'
                return

            plan_str = bundle.plan_str
            self.logger.info(f"Plan: {plan_str}")
            # self.plan_pub.publish(String(data=plan_str))  # for debugging visibility
            self.current_bundle = bundle
            self.current_plan   = plan_str
            self.current_names  = self._parse_plan_names(plan_str)

            # —— Pre-flight Coder: add any missing actions before ACT —— #
            have = self._function_names_from_file(ACTIONS_TMP_FILE if ACTIONS_TMP_FILE.exists() else ACTIONS_FILE)
            self.logger.info(f"Current actions in tmp: {sorted(have)} from {ACTIONS_TMP_FILE if ACTIONS_TMP_FILE.exists() else ACTIONS_FILE}")
            missing = self.current_names - have
            if missing:
                self.logger.info(f"Missing actions → coder: {sorted(missing)}")
                try:
                    res = self.coder.implement_actions(
                        actions       = missing,
                        pddl_schemas  = bundle.action_schemas,
                        plan_str      = plan_str,
                        agent_state   = {"scene_text": scene_text},
                        past_error_log= None
                    )
                except Exception as e:
                    self.logger.error(f"CoderLLM call failed: {e}")
                    self._after_coder_failure(group)
                    return

                if res.status != "ok":
                    self.logger.warn(f"CoderLLM merge/syntax failed: {res.status}")
                    self._after_coder_failure(group)
                    return

                # notify executor to hot-reload tmp actions
                self.code_pub.publish(String(data='reload'))

            # tell Referee which level to score
            self.ref_lvl_pub.publish(String(data=lvl_path))
            # Now dispatch to executor
            self.logger.info(f"Dispatching plan to executor: {meta['task_id']}")
            self.plan_pub.publish(String(data=plan_str)) 
            self.stage = 'wait_exec'
            return

        if self.stage == 'wait_exec':
            self.logger.info(f"WAIT_EXEC → {meta['task_id']}")

            self.logger.info(f"Task status: {self.task_status}, exec outcome: {self.exec_outcome}")

            # wait for referee result
            if self.task_status == 'running':
                return

            # stop executor regardless of result
            self.exec_cancel_pub.publish(String(data='cancel'))

            # ----- SUCCESS -----
            if self.task_status == 'success':
                self.logger.info("✅ success → commit tmp→main, next level")
                try:
                    if ACTIONS_TMP_FILE.exists():
                        shutil.copy2(ACTIONS_TMP_FILE, ACTIONS_FILE)
                except Exception as e:
                    self.logger.warn(f"Failed to commit tmp→main: {e}")
                # trust PDDL for this task group
                self.group_cache[lvl_entry['group']]["pddl"] = (
                    self.current_bundle.domain, self.current_bundle.problem
                )
                self.group_cache[lvl_entry['group']]["trusted"] = True

                # advance
                self.idx += 1
                self.stage = 'idle'
                return

            # ----- FAIL / TIMEOUT -----
            # Decide coder-retry vs re-plan using executor outcome
            exec_status = (self.exec_outcome or {}).get("status", "unknown")
            exec_trace  = (self.exec_outcome or {}).get("trace", "")
            # classify as coder-related?
            coder_related = exec_status in {"missing_method", "syntax_error", "runtime_error", "stuck"}

            if coder_related and self.retry_semantic < MAX_CODER_RETRIES:
                self.retry_semantic += 1
                self.logger.info(f"🔧 coder retry {self.retry_semantic}/{MAX_CODER_RETRIES} (executor={exec_status})")
                # Ask coder to (re)implement all actions referenced by the plan
                missing = self.current_names  # re-implement all to allow upgrades
                try:
                    res = self.coder.implement_actions(
                        actions       = missing,
                        pddl_schemas  = self.current_bundle.action_schemas,
                        plan_str      = self.current_plan,
                        agent_state   = {},
                        past_error_log= (self.exec_outcome or {}).get("msg","") + "\n" + exec_trace
                    )
                except Exception as e:
                    self.logger.error(f"CoderLLM call failed: {e}")
                    # fall through to re-plan
                    self._after_coder_failure(lvl_entry['group'])
                    return

                if res.status == "ok":
                    self.code_pub.publish(String(data='reload'))
                    # re-plan (state may have shifted) and try again
                    self.stage = 'plan'
                    return
                else:
                    self.logger.warn(f"CoderLLM merge/syntax failed: {res.status}")
                    self._after_coder_failure(lvl_entry['group'])
                    return

            # coder retries exhausted or not coder-related → re-plan
            self.logger.info("♻️ re-planning (coder exhausted or not applicable)")
            self.retry_semantic = 0
            self.spa_round += 1
            if self.spa_round > SPA_ROUNDS:
                self.logger.info("⛔ SPA rounds exhausted → skip level; reset tmp from main; clear group cache")
                # reset tmp from main
                try:
                    if ACTIONS_FILE.exists():
                        shutil.copy2(ACTIONS_FILE, ACTIONS_TMP_FILE)
                except Exception as e:
                    self.logger.warn(f"Failed to reset tmp from main: {e}")
                # clear group cache so next level starts fresh
                self.group_cache[lvl_entry['group']] = {"pddl": None, "trusted": False}
                # advance
                self.idx += 1
                self.stage = 'idle'
            else:
                # hint planner that previous plan failed
                self.group_cache[lvl_entry['group']]["trusted"] = False
                self.stage = 'plan'
            return

    # ---- small helper on coder failure during pre-flight ----
    def _after_coder_failure(self, group: str):
        self.retry_semantic = 0
        self.group_cache[group]["trusted"] = False
        self.spa_round += 1
        if self.spa_round > SPA_ROUNDS:
            self.logger.info("⛔ SPA rounds exhausted in pre-flight → skip level; reset tmp from main")
            try:
                if ACTIONS_FILE.exists():
                    shutil.copy2(ACTIONS_FILE, ACTIONS_TMP_FILE)
            except Exception as e:
                self.logger.warn(f"Failed to reset tmp from main: {e}")
            self.idx += 1
            self.stage = 'idle'
        else:
            self.stage = 'plan'


def main():
    rclpy.init()
    n = Supervisor()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
