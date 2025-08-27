#!/usr/bin/env python3
import os
import re
import json
import shutil
import yaml
import time
from pathlib import Path
import subprocess, threading
import signal

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Empty

from spca_llm_ur5.llm.plannerLLM import PlannerLLM
from spca_llm_ur5.llm.coderLLM   import CoderLLM
from spca_llm_ur5.llm.senseLLM   import SenseLLM

import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data 
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from spca_llm_ur5.scripts.runtime_paths import RUN_DIR, SEED_ACTS, BASE_ACTS, TMP_ACTS, RESET_SCRIPT, WORLD_FILE
ACTIONS_FILE     = BASE_ACTS
ACTIONS_TMP_FILE = TMP_ACTS
DOMAIN_FILE = RUN_DIR / "domain.pddl"
PROBLEM_FILE = RUN_DIR / "problem.pddl"

LAUNCH_CMD = ["ros2", "launch", "ur_yt_sim", "spawn_ur5_camera_gripper_moveit.launch.py",
            "world:=" + str(WORLD_FILE)]

SPA_ROUNDS = 5
MAX_CODER_RETRIES = 5

class Supervisor(Node):
    def __init__(self):
        super().__init__('supervisor')
        self.logger = self.get_logger()
        self.declare_parameter('curriculum_yaml', '')

        self.proc = subprocess.Popen(LAUNCH_CMD, preexec_fn=os.setsid)  # own process group
        time.sleep(10)  # give time for the process to start

        self.log_dir = RUN_DIR / "logs" / f"log_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        curr_path = self.get_parameter('curriculum_yaml').get_parameter_value().string_value
        self.levels = self._load_curriculum(curr_path)   # [{group, path, doc}]
        if not self.levels:
            self.logger.error("No levels found in curriculum.")
        # PDDL cache keyed by task_group
        self.group_cache = {}  # group -> {"pddl": (dom, prob) | None, "trusted": bool}

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability  = DurabilityPolicy.TRANSIENT_LOCAL

        # pubs/subs
        self.dispatch_pub = self.create_publisher(String, '/task/dispatch', qos)
        self.exec_cancel_pub   = self.create_publisher(String, '/executor/cancel', 10)

        self.ref_outcome = {'status': 'running', 'reason': ''}
        self.exec_outcome = {}
        self.ref_status = 'running'
        self.exec_status = 'unknown'
        self.create_subscription(String, '/task/status', self._task_cb, 10)
        self.create_subscription(String, '/executor/status', self._exec_cb, 10)

        # world reset
        self.reset_cli = self.create_client(Empty, '/reset_world')
        # self.reset_cli = self.create_client(Empty, '/reset_simulation')
        self.reset_world_in_progress = False

        # loop state
        self.idx = 0
        self.stage = 'idle'
        self.retry_semantic = 0
        self.spa_round = 1

        self.coder_err_log = ""

        # wait for Referee
        self._exec_success_t = None
        self._ref_grace_s = 5.0

        self.planner = PlannerLLM()
        self.coder   = CoderLLM()
        self.sense   = SenseLLM()

        self._bridge = CvBridge()
        self.last_image_bgr = None
        self.create_subscription(
            Image,
            "/camera/image_raw",
            self._image_cb,
            qos_profile=10#qos_profile_sensor_data
        )

        self.scene_text = "" 
        
        self.timer = self.create_timer(0.1, self._tick)

        checkpoint_level = "cube_near" #"levels/touch/level_07_cube_right.yaml"
        reuse_agent_actions = True # reuse actions.py from previous run
        reuse_pddls = True  # reuse PDDL files from previous run
        self._load_checkpoints(checkpoint_level, reuse_agent_actions, reuse_pddls)

    # -------- helpers ----------
    def _task_cb(self, msg: String):
        try:
            self.ref_outcome = json.loads(msg.data)
        except Exception:
            # legacy plain string → wrap it
            self.ref_outcome = {"status": msg.data, "reason": ""}

    def _exec_cb(self, msg: String):
        try:
            self.exec_outcome = json.loads(msg.data)
        except Exception:
            self.exec_outcome = {"status": "unknown", "msg": msg.data, "trace": ""}
    
    def _hard_reset(self):
        # 1) graceful stop
        os.killpg(self.proc.pid, signal.SIGINT)  # Ctrl-C to ros2 launch
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)

        # 2) (optional) hard cleanup if Gazebo hung
        os.system("pkill -f gzserver || true; pkill -f gzclient || true")

        # 3) relaunch fresh
        self.logger.info("Supervisor: wating before relaunch...")
        time.sleep(5)  # tiny breather so ports/shm clear
        self.proc = subprocess.Popen(LAUNCH_CMD, preexec_fn=os.setsid)
        self.logger.info("Supervisor: relaunched Gazebo with UR5, camera, and gripper.")
        time.sleep(1)  # give time for the process to start
        self._reset_world_async()
        self.stage = 'resetting' 

    # def _reset_world_async(self):
    #     if self.reset_world_in_progress:
    #         return
    #     self.reset_world_in_progress = True
    #     self.stage = "resetting"            # wait in _tick

    #     def _runner():
    #         self.get_logger().info(f"🧹 RESET - running {RESET_SCRIPT} …")
    #         proc = subprocess.run(
    #             ["/bin/bash", str(RESET_SCRIPT)],
    #             capture_output=True, text=True
    #         )
    #         # Hand the result back to the rclpy thread:
    #         rclpy.get_default_context().call_soon_threadsafe(
    #             self._on_reset_finished, proc
    #         )

    #     threading.Thread(target=_runner, daemon=True).start()

    # def _on_reset_finished(self, proc: subprocess.CompletedProcess):
    #     self.reset_world_in_progress = False
    #     if proc.returncode == 0:
    #         self.get_logger().info("✅ RESET script finished OK")
    #         # clear transient state exactly as before
    #         self.ref_outcome = {"status": "running", "reason": ""}
    #         self.exec_outcome = {}
    #         self.retry_semantic = 0
    #         self.spa_round = 1
    #         self.stage = "sense"             # carry on
    #     else:
    #         self.get_logger().error(
    #             f"❌ RESET script failed ({proc.returncode}):\n{proc.stderr}"
    #         )
    #         self.stage = "idle"              # try again / skip

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
            self.ref_outcome = {'status': 'running', 'reason': ''}
            self.exec_outcome = {}
            self.retry_semantic = 0
            self.spa_round = 1
            self.stage = 'sense'
        except Exception as e:
            self.logger.error(f"Reset world service call failed: {e}")
            # Handle error: perhaps retry or skip level
            self.stage = 'idle' # Revert to idle to try again
    
    # def wait_for_services(self):
    #     self.logger.info("Waiting for core services...")
    #     self.cart_cli.wait_for_service()
    #     self.follow_traj_ac.wait_for_server()
    #     self.gripper_ac.wait_for_server()
    #     self.logger.info("All core services and actions are ready.")

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
    
    def _load_checkpoints(self, checkpoint_level: str = None, reuse_agent_actions: bool = False, reuse_pddls: bool = False):
        """
        Start simulation from a specified level file.
        If reuse_agent_actions is True, it will reuse the actions.py file.
        If reuse_pddls is True, it will reuse PDDL files from previous runs.
        """
        if checkpoint_level:
            self.logger.info(f"Loading checkpoint level: {checkpoint_level}")
            for i, lvl in enumerate(self.levels):
                if checkpoint_level in lvl['path']:
                    self.idx = i
                    break
        # else: self.idx = 0  

        # Load agent actions
        if reuse_agent_actions and ACTIONS_TMP_FILE.exists():
            # do nothing
            self.logger.info(f"Reusing agent actions from {ACTIONS_TMP_FILE}")
        else: # reset to seed
            shutil.copy(SEED_ACTS, ACTIONS_FILE)
            shutil.copy(SEED_ACTS, ACTIONS_TMP_FILE)
            self.logger.info(f"Using seed agent actions from {SEED_ACTS}")

        # Load PDDL files (only current level) - cache for the group
        if reuse_pddls:
            if DOMAIN_FILE.exists() and PROBLEM_FILE.exists():
                self.logger.info(f"Reusing PDDL files: {DOMAIN_FILE}, {PROBLEM_FILE}")
                group = self.levels[self.idx]["group"]
                self._ensure_group_cache(group)
                dom = DOMAIN_FILE.read_text()
                prob = PROBLEM_FILE.read_text()
                self.group_cache[group]["pddl"] = (dom, prob)
                self.group_cache[group]["trusted"] = True
            else:
                self.logger.warn("PDDL files not found; will regenerate them.")
        else:
            # ignore
            pass
    
    def _save_checkpoints_log(self):
        """Save the current state of the supervisor to a log file."""
        # random suffix
        identifier= f"{self.idx}_{time.strftime('%H%M%S')}"
        log_path = self.log_dir / f"checkpoint_{identifier}.log"
        with open(log_path, 'w') as f:
            f.write(f"Current level index: {self.idx}\n")
            f.write(f"Stage: {self.stage}\n")
            f.write(f"Retry semantic: {self.retry_semantic}\n")
            f.write(f"SPA round: {self.spa_round}\n")
            f.write(f"Referee outcome: {json.dumps(self.ref_outcome)}\n")
            f.write(f"Executor outcome: {json.dumps(self.exec_outcome)}\n")
        self.logger.info(f"Checkpoint saved to {log_path}")

        # Copy agent actions and PDDL files to the log directory
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Copying checkpoint files to {self.log_dir}...")
        if not ACTIONS_FILE.exists() or not ACTIONS_TMP_FILE.exists():
            self.logger.warn("Agent actions files do not exist; skipping copy.")
        else:
            shutil.copy(ACTIONS_FILE, self.log_dir / f"agent_actions_{identifier}.py")
            shutil.copy(ACTIONS_TMP_FILE, self.log_dir / f"agent_actions_tmp_{identifier}.py")
        if not RUN_DIR.exists():
            self.logger.warn("Run directory does not exist; skipping PDDL copy.")
        elif not DOMAIN_FILE.exists() or not PROBLEM_FILE.exists():
            self.logger.warn("PDDL files do not exist; skipping copy.")
        else:
            shutil.copy(DOMAIN_FILE, self.log_dir / f"domain_{identifier}.pddl")
            shutil.copy(PROBLEM_FILE, self.log_dir / f"problem_{identifier}.pddl")
        # shutil.copy(ACTIONS_FILE, self.log_dir / f"agent_actions_{identifier}.py")
        # shutil.copy(ACTIONS_TMP_FILE, self.log_dir / f"agent_actions_tmp_{identifier}.py")
        # shutil.copy(RUN_DIR / "domain.pddl", self.log_dir / f"domain_{identifier}.pddl")
        # shutil.copy(RUN_DIR / "problem.pddl", self.log_dir / f"problem_{identifier}.pddl")
        # self.logger.info("Checkpoint files copied to log directory.")

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
            rclpy.spin_once(self, timeout_sec=0.5)
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

    # self.stage ∈ {"idle", "resetting", "sense", "plan", "code", "wait_exec"}
    def _tick(self):
        # 0. finished?
        if self.idx >= len(self.levels):
            return
        
        self.logger.info(f"🔄 SPA TICK {self.idx}/31 ({self.levels[self.idx]['doc']['task_id']}) - " \
                         f"stage: {self.stage}, " \
                         f"round: {self.spa_round}")
        
        self._ensure_group_cache(self.levels[self.idx]["group"])

        # 1. ------------------------------------------------ idle ------------
        if self.stage == "idle":
            # reset world → async future; when done we’ll enter "sense"
            self._reset_world_async()
            self.stage = "resetting"
            return
        
        # 1.1 ------------------------------------------ resetting -----------
        if self.stage == "resetting": return # wait until _reset_world_response_cb flips stage

        # 2. --------------------------------------------- sense -------------
        if self.stage == "sense":
            self.logger.info("👁️  SENSE - capturing scene and describing environment…")
            time.sleep(5)  # wait for Gazebo to settle
            img = self._get_img(wait_s=10.0)            # blocking ≤10 s
            self.scene_text = self.sense.describe(      # may return ""
                title=self.levels[self.idx]["doc"].get("title", ""),
                description=self.levels[self.idx]["doc"].get("description", ""),
                bgr_image=img)
            # self.scene_text =  'The image shows a workspace with objects arranged in a row. In the top row, there are four colored circles: blue, red, green, and another red. In the bottom row, there are three colored shapes: a green square on the left, a blue cube in the middle, and a red cube on the right. The gripper is positioned near the bottom of the image but is not touching any objects. The objective is to move the gripper to make contact with the blue cube in the middle of the bottom row.'
            self.logger.info(f"Scene text: {self.scene_text[:100]}...")  # log first 100 chars
            self.stage = "plan"
            return
        
        if self.stage == "sense_code":
            self.logger.info("👁️  CODE SENSE - capturing scene and describing environment…")
            img = self._get_img(wait_s=10.0)            # blocking ≤10 s
            self.scene_text = self.sense.describe(      # may return ""
                title=self.levels[self.idx]["doc"].get("title", ""),
                description=self.levels[self.idx]["doc"].get("description", ""),
                bgr_image=img)
            # self.scene_text =  'The image shows a workspace with objects arranged in a row. In the top row, there are four colored circles: blue, red, green, and another red. In the bottom row, there are three colored shapes: a green square on the left, a blue cube in the middle, and a red cube on the right. The gripper is positioned near the bottom of the image but is not touching any objects. The objective is to move the gripper to make contact with the blue cube in the middle of the bottom row.'
            self.logger.info(f"Scene text: {self.scene_text[:100]}...")  # log first 100 chars
            self.stage = "code"
            return

        # 3. ---------------------------------------------- plan -------------
        if self.stage == "plan":
            self.logger.info(f"🧩 PLAN  (SPA {self.spa_round}) - asking LLM for PDDL + plan…")
            cache = self.group_cache[self.levels[self.idx]["group"]]
            try:
                self.bundle = self.planner.plan(
                    snapshot={"scene_text": self.scene_text},
                    meta     = {
                        "task_group": self.levels[self.idx]["group"],
                        "task_id": self.levels[self.idx]["doc"].get("task_id", Path(self.levels[self.idx]["path"]).stem),
                        "title": self.levels[self.idx]["doc"].get("title", ""),
                        "description": self.levels[self.idx]["doc"].get("description", ""),
                    },
                    pddl_hint=cache["pddl"],
                    pddl_trusted=cache["trusted"],
                    plan_failed=(self.retry_semantic >= MAX_CODER_RETRIES)
                )
                self.plan_str  = self.bundle.plan_str
                self.logger.info(f"📜 PLAN: {self.plan_str}")
                self.act_names = self._parse_plan_names(self.plan_str)
                self.stage = "code"
            except Exception:                # PDDL syntax or LLM failure
                self.bundle = None
                self.plan_str=""
                self.act_names=set()
                self._bump_spa_round(plan_failed=True)
            return

        # 4. ----------------------------------------------- code ------------
        if self.stage == "code":
            self.logger.info("🛠️  CODE - (re)generating / hot-reloading missing actions…")
            have = self._function_names_from_file(ACTIONS_TMP_FILE if ACTIONS_TMP_FILE.exists()
                                            else ACTIONS_FILE)
            missing = self.act_names - have
            if self.coder_err_log:
                missing = self.act_names  # re-implement all to allow upgrades

            if missing:
                self.logger.info(f"Missing actions → coder: {sorted(missing)}")
                res = self.coder.implement_actions(
                        actions=missing,
                        pddl_schemas=self.bundle.action_schemas,
                        plan_str=self.plan_str,
                        agent_state={"scene_text": self.scene_text},
                        past_error_log=self.coder_err_log)
                # res = type('Response', (object,), {'status': "ok"})()
                self.logger.info(f"Coder result: {res}")
                if res.status != "ok":              # merge/syntax failed
                    self._bump_spa_round(coder_failed=True)
                    return

            self.coder_err_log = ""  # reset error log
            # nothing missing or coder succeeded → execute

            # start executor with plan, and send level to referee)
            lvl_path = self.levels[self.idx]["path"]
            edit_plan = self.plan_str
            # edit_plan = "[gripper_open()]"#, " + self.plan_str[1:]
            # edit_plan = "[gripper_open(), move_to_predefined()]"
            payload = json.dumps({
                "level_path": lvl_path,
                "plan":       edit_plan,
            })
            self.dispatch_pub.publish(String(data=payload))

            self.stage = "wait_exec"
            return

        # 5. ------------------------------------------- wait_exec -----------
        if self.stage == "wait_exec":
            # # update cached statuses if new messages arrived
            # if self.exec_outcome:                # /executor/outcome
            #     self.exec_status = self.exec_outcome.get("status", "unknown")
            # if self.ref_outcome:                 # /task/status
            #     self.ref_status = self.ref_outcome.get("status", "unknown")

            self.exec_status = self.exec_outcome.get("status","unknown") if self.exec_outcome else "unknown"
            self.ref_status = self.ref_outcome .get("status","running")

            # # still waiting for either side?
            # if self.exec_status is None or self.ref_status is None:
            #     return
            # we do not need both sides (if one fails, throw imidiately, dont wait for the other)
            
            # if self.ref_status == "running":
            #     self.logger.info("Referee still running; waiting...")
            #     return
            # what if executor fails? or even finishes successfully? 
            # - just ref running will ignore both and we will have to wait for timeout

            # ----- SUCCESS path --------------------------------------------
            if self.ref_status == "success":
                self.logger.info("🏆 REFEREE - ✅ level success! committing code & advancing…")
                self._commit_tmp_to_main()
                self._trust_pddl_for_group()
                self._save_checkpoints_log()
                self.dispatch_pub.publish(String(data=json.dumps({
                    "level_path": None,
                    "plan": None,
                })))
                self.idx   += 1                   # next level
                self.stage  = "idle"
                self.spa_round = 1                # reset round
                self.ref_outcome = {'status': 'running', 'reason': ''}
                self.exec_outcome = {}
                self.retry_semantic = 0
                self._exec_success_t = None
                self._hard_reset()  # hard reset world
                return
            
            if self.ref_status == "running" and self.exec_status == "unknown":
                # self.logger.info("Referee still running; waiting...")
                return
            
            # if self.ref_status in ("fail", "timeout") 
            #     go to error handling
            # if self.exec_status not in ("success", "unknown"):
            #     go to error handling
            
            # ----- FAILURE / RETRY ----------------------------------------
            
            if self.exec_status == "success" and self.ref_status == "running":
                if self._exec_success_t is None:
                    self._exec_success_t = time.monotonic()
                # give the referee a moment to see final contacts
                if time.monotonic() - self._exec_success_t < self._ref_grace_s:
                    return
                # after grace, if ref still hasn't judged, fall through to retry logic

            # if self.exec_status == "success":
            #     # Executed all actions successfully, but referee did not confirm success - we did not reach the goal state
            #     self.logger.warn("Executor success but referee did not confirm success.")
            #     # re-plan
            #     self.retry_semantic = 0
            #     self._bump_spa_round(plan_failed=True)
            #     return
            # if exec success, can also go into coder semantic retry
            
            if self.retry_semantic < MAX_CODER_RETRIES:
                self.logger.info(f"🔄 SEMANTIC RETRY {self.retry_semantic}/{MAX_CODER_RETRIES} - refining code on same plan…")
                self.coder_err_log = ""
                
                if self.ref_status in ("fail", "timeout"):
                    self.logger.warn(f"❌ REFEREE - {self.ref_status} level outcome.")
                    # self.exec_cancel_pub.publish(String(data=json.dumps({"status": "cancelled"})))
                    self.coder_err_log = "Referee outcome:\n" + json.dumps(self.ref_outcome, indent=2)
                    self._hard_reset()  # hard reset world

                if self.exec_status not in ("success", "unknown"):
                    self.logger.warn(f"❌ EXECUTOR - {self.exec_status} level outcome.")
                    self.coder_err_log += "\nExecutor outcome:\n" + json.dumps(self.exec_outcome, indent=2)

                if not self.coder_err_log and self.exec_status == "success":
                    self.logger.warn("Executor reported success, but referee did not confirm success.")
                    self.coder_err_log = "All actions executed successfully, but referee did not confirm success."

                self.logger.info(f"Error log: {self.coder_err_log}")
                self.retry_semantic += 1
                # self.stage = "code"               # re-run coder on same plan
                self.stage = "sense_code" 

                # stop referee and executor
                # self.exec_cancel_pub.publish(String(data=json.dumps({"status": "cancelled"})))
                self.dispatch_pub.publish(String(data=json.dumps({
                    "level_path": None,
                    "plan":       None,
                })))
                # reset outcomes
                self.ref_outcome = {'status': 'running', 'reason': ''}
                self.exec_outcome = {}
                self._exec_success_t = None
                return

            # else: re-plan (semantic mismatch or coder exhausted)
            self.retry_semantic = 0
            self._bump_spa_round(plan_failed=True)
            return
        
    def _bump_spa_round(self, *, coder_failed=False, plan_failed=False):
        self.logger.info(f"♻️  SPA ROUND {self.spa_round}/{SPA_ROUNDS} - starting fresh sense→plan cycle…")
        self.group_cache[self.levels[self.idx]["group"]]["trusted"] = False
        if plan_failed or coder_failed:
            self.spa_round += 1
            if self.spa_round > SPA_ROUNDS:
                # skip level, reset main→tmp etc.
                self._skip_level()
            else:
                self.stage = "sense"          # fresh SENSE → PLAN cycle
    
    def _skip_level(self):
        self.logger.warn(f"⛔ Skipping level {self.levels[self.idx]['path']} after {self.spa_round} rounds.")
        # reset tmp (copy main to tmp)
        try:
            shutil.copy2(ACTIONS_FILE, ACTIONS_TMP_FILE)
        except Exception as e:
            self.logger.error(f"Failed to reset tmp actions: {e}")
        # reset world and counters
        self.dispatch_pub.publish(String(data=json.dumps({
            "level_path": None,
            "plan": None,
        })))
        self.coder_err_log=""
        self.exec_outcome = {}
        self.ref_outcome = {'status':'running','reason':''}
        self.stage = 'idle'
        self.idx += 1
        self.spa_round = 1
        self.retry_semantic = 0
        self._exec_success_t = None
        self._hard_reset()  # hard reset world

    def _commit_tmp_to_main(self):
        """Commit tmp actions to main."""
        try:
            if ACTIONS_TMP_FILE.exists():
                shutil.copy2(ACTIONS_TMP_FILE, ACTIONS_FILE)
                self.logger.info(f"Committed actions from {ACTIONS_TMP_FILE} to {ACTIONS_FILE}.")
            else:
                self.logger.warn(f"No tmp actions file found at {ACTIONS_TMP_FILE}.")
        except Exception as e:
            self.logger.error(f"Failed to commit tmp→main: {e}")
    
    def _trust_pddl_for_group(self):
        """Trust PDDL for the current task group."""
        group = self.levels[self.idx]["group"]
        self.group_cache[group]["trusted"] = True
        self.group_cache[group]["pddl"] = (
            self.bundle.domain, self.bundle.problem
        )
        self.logger.info(f"Trusted PDDL for group {group}.")


def main():
    rclpy.init()
    n = Supervisor()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
