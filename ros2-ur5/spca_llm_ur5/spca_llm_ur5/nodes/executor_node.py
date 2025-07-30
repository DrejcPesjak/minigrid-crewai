#!/usr/bin/env python3
# executor_node.py
import re, traceback
from pathlib import Path
import importlib.util

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from spca_llm_ur5.nodes.ctx_runtime import Ctx
from spca_llm_ur5.runtime_paths import BASE_ACTS, TMP_ACTS
ACTIONS_FILE     = BASE_ACTS
ACTIONS_TMP_FILE = TMP_ACTS

def parse_plan(plan_str: str):
    """
    Accepts either:
      - "[move-forward(), pick-up(box_blue)]"  (PDDL→UP→list form)
      - "move_forward(); gripper_open()"       (semicolon form)
    Returns: list[(name:str, args:list[str])] with names kebab→snake.
    Only bare identifier args allowed.
    """
    s = plan_str.strip()
    steps = []

    def _parse_call(text: str):
        m = re.match(r'^([A-Za-z][\w\-]*)\s*\(\s*([^)]*?)\s*\)$', text)
        if not m:
            raise ValueError(f'Bad step: {text}')
        name = m.group(1).replace('-', '_')
        argstr = m.group(2).strip()
        args = [a.strip() for a in argstr.split(',')] if argstr else []
        for a in args:
            if not re.match(r'^[A-Za-z_]\w*$', a):
                raise ValueError(f'Bad arg token: {a}')
        return name, args

    if s.startswith('[') and s.endswith(']'):
        body = s[1:-1].strip()
        tokens = [t.strip() for t in re.split(r'\)\s*,', body) if t.strip()]
        for t in tokens:
            if not t.endswith(')'): t += ')'
            steps.append(_parse_call(t))
        return steps

    parts = [p.strip() for p in s.split(';') if p.strip()]
    for p in parts:
        steps.append(_parse_call(p))
    return steps


class Executor(Node):
    def __init__(self):
        super().__init__('executor')

        # runtime context (MoveIt, TF, camera, cloud, actions/services)
        self.ctx = Ctx(self)

        # load actions module from disk (tmp preferred)
        self.agent = None
        self._load_actions_initial()

        # topics
        self.create_subscription(String, '/planner/plan', self._plan_cb, 10)
        # TRIGGER only: when any message arrives, reload from disk
        self.create_subscription(String, '/coder/script', self._reload_trigger_cb, 10)
        self.create_subscription(String, '/executor/cancel', self._cancel_cb, 10)
        self.status_pub = self.create_publisher(String, '/executor/status', 10)

    # ---------- module loading ----------
    def _load_from_file(self, path: Path):
        spec = importlib.util.spec_from_file_location('agent_actions', str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _choose_actions_path(self) -> Path | None:
        if ACTIONS_TMP_FILE.exists():
            return ACTIONS_TMP_FILE
        if ACTIONS_FILE.exists():
            return ACTIONS_FILE
        return None

    def _load_actions_initial(self):
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

    def _reload_trigger_cb(self, _msg: String):
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
    def _plan_cb(self, msg: String):
        self.ctx.clear_cancel()
        try:
            for name, args in parse_plan(msg.data):
                fn = getattr(self.agent, name, None)
                if not callable(fn):
                    raise RuntimeError(f"missing action: {name}")
                # Log function name and arguments
                self.get_logger().info(f"Executing action: {name} with arguments: {args}")
                # ACTION signature: ctx + string args only
                fn(self.ctx, *args)
            self.status_pub.publish(String(data='success'))
        except Exception:
            self.get_logger().error(traceback.format_exc())
            self.status_pub.publish(String(data='fail'))


def main():
    rclpy.init()
    node = Executor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
