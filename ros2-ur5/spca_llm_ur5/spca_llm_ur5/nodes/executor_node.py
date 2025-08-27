#!/usr/bin/env python3
# executor_node.py
import time
import re, traceback, json
from pathlib import Path
import importlib.util

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
# from rclpy.logging import LoggingSeverity, set_logger_level

from spca_llm_ur5.nodes.ctx_runtime import Ctx
from spca_llm_ur5.scripts.runtime_paths import BASE_ACTS, TMP_ACTS
ACTIONS_FILE     = BASE_ACTS
ACTIONS_TMP_FILE = TMP_ACTS

_action_re = re.compile(r'([a-zA-Z][\w-]*)\s*\(\s*([^)]*?)\s*\)')

def parse_plan(plan: str):
    """
    Expects UP bracket form only, e.g.
    "[move-forward(), pick-up(box_blue)]"
    Returns list[(name, args)]
    """
    result = []
    for m in _action_re.finditer(plan):
        name = m.group(1).replace('-', '_')
        args = [a.strip() for a in m.group(2).split(',')] if m.group(2) else []
        result.append((name, args))
    return result

class Executor(Node):
    def __init__(self):
        super().__init__('executor')

        # set_logger_level('tf2_buffer', LoggingSeverity.ERROR)
        # set_logger_level('tf2_ros',   LoggingSeverity.ERROR)

        # runtime context (MoveIt, TF, camera, cloud, actions/services)
        self.ctx = Ctx(self)

        # load actions module from disk (tmp preferred)
        self.agent = None
        # self._load_actions_initial()

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability  = DurabilityPolicy.TRANSIENT_LOCAL

        # topics
        self.create_subscription(String, '/task/dispatch', self._execute_cb, qos_profile=qos)
        self.create_subscription(String, '/executor/cancel', self._cancel_cb, 10)
        self.status_pub = self.create_publisher(String, '/executor/status', 10)

        self._plan_queue = []
        self._execution_in_progress = False
        self._current_action = ""
        self._cancellation_requested = False

    # ---------- module loading ----------
    def _load_from_file(self, path: Path):
        spec = importlib.util.spec_from_file_location('agent_actions', str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # sanity check, print all functions
        funcs = [f for f in dir(mod) if callable(getattr(mod, f)) and not f.startswith('_')]
        self.get_logger().info(f"DEBUG:Loaded actions module from {path}: {len(funcs)} functions: {', '.join(funcs)}")
        return mod

    def _choose_actions_path(self) -> Path | None:
        if ACTIONS_TMP_FILE.exists():
            return ACTIONS_TMP_FILE
        if ACTIONS_FILE.exists():
            return ACTIONS_FILE
        return None

    def _load_actions_initial(self):
        self.get_logger().info("DEBUG:Loading actions module from disk...")
        path = self._choose_actions_path()
        if path:
            try:
                self.agent = self._load_from_file(path)
            except Exception:
                self.get_logger().error(traceback.format_exc())
        if self.agent is None:
            # start empty module if nothing on disk yet
            spec = importlib.util.spec_from_loader('agent_actions', loader=None)
            self.agent = importlib.util.module_from_spec(spec)
            exec("# empty", self.agent.__dict__)

    def _reload_agent(self):
        self.get_logger().info("DEBUG:Reloading actions module from disk...")
        path = self._choose_actions_path()
        if not path:
            return
        try:
            self.agent = self._load_from_file(path)
        except Exception:
            self.get_logger().error(traceback.format_exc())

    def _cancel_cb(self, _msg: String):
        self.get_logger().info("Cancellation requested. Stopping current plan.")
        self._cancellation_requested = True

    # ---------- plan execution (revised) ----------
    def _execute_cb(self, msg: String):
        self.get_logger().info(f"DEBUG:Received plan: {msg.data}")
        plan_str = json.loads(msg.data).get("plan", "")
        if not plan_str:
            self.get_logger().warn("Received empty plan; nothing to execute.")
            # self._send_outcome(success=True) # or a neutral status
            return
        
        # # wait until TF/MoveIt/joints are sane post-reset
        # if not self.ctx.ensure_ready(timeout=8.0):
        #     self.get_logger().warn("Executor not ready after reset; deferring plan.")
        #     # self.status_pub.publish(String(data=json.dumps({"status":"reset_wait"})))
        #     return
        
        self.ctx.clear_cancel()
        self._reload_agent()

        # Start the execution by parsing the plan and putting it into a queue
        self._plan_queue = parse_plan(plan_str)
        self._execution_in_progress = True
        self.get_logger().info(f"Starting execution of {len(self._plan_queue)} actions.")
        self._run_next_action()

    def _run_next_action(self):
        if self._cancellation_requested:
            self.get_logger().info("Plan cancelled.")
            self._send_outcome(success=False, msg="Plan cancelled by user.")
            self._execution_in_progress = False
            self._cancellation_requested = False # Reset the flag
            return
        # If the queue is empty, we are done
        if not self._plan_queue:
            self.get_logger().info("All actions in plan completed.")
            self._send_outcome(success=True)
            self._execution_in_progress = False
            return
        
        # Get the next action from the queue
        name, args = self._plan_queue.pop(0)
        self._current_action = name

        self.get_logger().info(f"---- Executing action -> {name}({', '.join(args)})")

        fn = getattr(self.agent, name, None)
            
        if not callable(fn):
            self.get_logger().error(f"Missing action: {name}")
            self._send_outcome(success=False, msg=f"Missing action: {name}")
            self._execution_in_progress = False
            return

        try:
            # Call the action with a callback to be notified upon completion
            fn(self.ctx, *args, done_callback=self._action_done_cb)

        except Exception as exc:
            m = f"Error executing action {name} with args {args}: {exc}"
            t = traceback.format_exc()
            self.get_logger().error(m)
            self.get_logger().error(t)
            self._send_outcome(success=False, msg=m, trace=t)
            self._execution_in_progress = False

    def _action_done_cb(self, success, msg="", trace=""):
        """
        This callback is triggered by an action when it finishes.
        """
        if success:
            self.get_logger().info(f"Action completed successfully. Running next action.")
            # Run the next action in the queue
            self._run_next_action()
        else:
            self.get_logger().error(f"Action failed with message: {msg}")
            m = f"Action {self._current_action} failed: {msg}"
            self._send_outcome(success=False, msg=m, trace=trace)
            self._execution_in_progress = False

    def _send_outcome(self, success: bool, msg: str = "", trace: str = ""):
        """
        Sends the final outcome message to the /executor/status topic.
        """
        if success:
            outcome = {"status": "success", "msg": msg, "trace": trace}
        else:
            outcome = {"status": "failed", "msg": msg, "trace": trace}

        self.get_logger().info(f"Executor outcome: {outcome['status']}")
        self.status_pub.publish(String(data=json.dumps(outcome)))

def main():
    rclpy.init()
    node = Executor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
