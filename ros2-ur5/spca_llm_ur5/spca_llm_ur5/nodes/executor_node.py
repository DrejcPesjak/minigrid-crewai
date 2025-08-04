#!/usr/bin/env python3
# executor_node.py
import time
import re, traceback, json
from pathlib import Path
import importlib.util

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

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

        # runtime context (MoveIt, TF, camera, cloud, actions/services)
        self.ctx = Ctx(self)

        # load actions module from disk (tmp preferred)
        self.agent = None
        # self._load_actions_initial()

        # topics
        self.create_subscription(String, '/task/dispatch', self._execute_cb, 10)
        self.create_subscription(String, '/executor/cancel', self._cancel_cb, 10)
        self.status_pub = self.create_publisher(String, '/executor/status', 10)

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
        self.ctx.cancel()

    # ---------- plan execution ----------
    def _execute_cb(self, msg: String):
        self.get_logger().info(f"DEBUG:Received plan: {msg.data}")
        plan_str = json.loads(msg.data).get("plan", "")
        if not plan_str:
            self.get_logger().warn("Received empty plan; nothing to execute.")
            return
        
        self.ctx.clear_cancel()
        try:
            self._reload_agent()
            time.sleep(1.0)  # give time for the module to reload

            for name, args in parse_plan(plan_str):
                self.get_logger().info(f"---- Executing action -> {name}({', '.join(args)})")

                fn = getattr(self.agent, name, None)
                    
                if not callable(fn):
                    raise RuntimeError(f"missing action: {name}")
                
                fn(self.ctx, *args)
                time.sleep(5.0)
            outcome = {"status": "success", "msg": "", "trace": ""}
        except Exception as exc:
            outcome = {
                "status": type(exc).__name__,
                "msg": str(exc),
                "trace": traceback.format_exc(),
            }

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
