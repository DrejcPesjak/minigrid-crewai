#!/usr/bin/env python3
import rclpy, yaml, threading, subprocess, re, time
from rclpy.node import Node
from std_msgs.msg import String

PAT_C1 = re.compile(r'collision1:\s*"([^"]+)"')
PAT_C2 = re.compile(r'collision2:\s*"([^"]+)"')

def _model_of(s: str) -> str: return s.split('::', 1)[0]
def _key(a, b): return tuple(sorted((a, b)))

class Referee(Node):
    def __init__(self):
        super().__init__('referee')
        self.declare_parameter('level_yaml', '')
        self.declare_parameter('contacts_topic', '/gazebo/default/physics/contacts')
        self.declare_parameter('ignore_models', [])          # default: do NOT ignore ground_plane
        self.declare_parameter('contact_stale_s', 0.3)       # TTL for contact pairs (s)

        self.level = None
        self.start_t = time.time()

        # contacts: last-seen timestamp per pair
        self._last_seen = {}       # {(a,b): t_monotonic}
        self._lock = threading.Lock()

        lvl = self.get_parameter('level_yaml').get_parameter_value().string_value
        if lvl: self._load_level(lvl)

        topic = self.get_parameter('contacts_topic').get_parameter_value().string_value
        threading.Thread(target=self._reader, args=(topic,), daemon=True).start()

        self.pub = self.create_publisher(String, '/task/status', 10)
        self.timer = self.create_timer(0.2, self._tick)
        self.setlvl_sub = self.create_subscription(String, '/referee/set_level', self._set_level, 10)

    def _set_level(self, msg:String):
        self._load_level(msg.data)

    def _load_level(self, path:str):
        with open(path, 'r') as f:
            self.level = yaml.safe_load(f)
        self.start_t = time.time()
        self.get_logger().info(f"Loaded level {self.level['task_id']}")# from {path}")

    def _reader(self, topic: str):
        proc = subprocess.Popen(['gz', 'topic', '-e', topic],
                                text=True, stdout=subprocess.PIPE, bufsize=1)
        cur_c1 = None
        ignore = set(self.get_parameter('ignore_models').get_parameter_value().string_array_value
                     or self.get_parameter('ignore_models').value)
        for line in proc.stdout:
            m1, m2 = PAT_C1.search(line), PAT_C2.search(line)
            if m1:
                cur_c1 = m1.group(1)
            if m2 and cur_c1:
                c1, c2 = _model_of(cur_c1), _model_of(m2.group(1))
                if c1 in ignore or c2 in ignore:
                    cur_c1 = None
                    continue
                with self._lock:
                    self._last_seen[_key(c1, c2)] = time.monotonic()
                cur_c1 = None

    def _snapshot_pairs(self):
        ttl = float(self.get_parameter('contact_stale_s').value)
        now = time.monotonic()
        with self._lock:
            # prune old entries and return current set
            stale = [k for k, t in self._last_seen.items() if (now - t) > ttl]
            for k in stale:
                self._last_seen.pop(k, None)
            return set(self._last_seen.keys())

    def _require_pairs(self, want_pairs, should_exist: bool) -> bool:
        cur = self._snapshot_pairs()
        for a, b in want_pairs or []:
            exists = _key(a, b) in cur
            if should_exist and not exists: return False
            if not should_exist and exists: return False
        return True

    def _tick(self):
        if not self.level:
            self.pub.publish(String(data='running'))
            return
        if (time.time() - self.start_t) > float(self.level.get('time_limit_s', 120)):
            self.pub.publish(String(data='timeout')); return

        succ = self.level['success']
        fail = self.level.get('fail', {'collisions_true': [], 'collisions_false': []})

        # Fail first
        if not self._require_pairs(fail.get('collisions_false', []), should_exist=False):
            self.pub.publish(String(data='fail')); return
        if not self._require_pairs(fail.get('collisions_true', []), should_exist=True):
            self.pub.publish(String(data='fail')); return

        ok_true  = self._require_pairs(succ.get('collisions_true', []),  True)
        ok_false = self._require_pairs(succ.get('collisions_false', []), False)
        self.pub.publish(String(data='success' if (ok_true and ok_false) else 'running'))

def main():
    rclpy.init()
    n = Referee()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
