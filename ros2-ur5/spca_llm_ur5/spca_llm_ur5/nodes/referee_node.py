#!/usr/bin/env python3
import rclpy, yaml, threading, subprocess, re, time, json
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

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

        self._gz_proc = None
        self._last_contact_line_t = 0.0
        self._last_contacts = set()  # last seen pairs (a,b) at the last tick
        topic = self.get_parameter('contacts_topic').get_parameter_value().string_value
        threading.Thread(target=self._reader, args=(topic,), daemon=True).start()

        self.get_logger().info(f"Referee past reader")

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability  = DurabilityPolicy.TRANSIENT_LOCAL

        self.pub = self.create_publisher(String, '/task/status', 10)
        self.timer = self.create_timer(0.2, self._tick)
        self.create_subscription(String, '/task/dispatch', self._set_level, qos_profile=qos)
        self.get_logger().info(f"Referee node initialized")

    def _set_level(self, msg:String):
        self.get_logger().info(f"Referee: received level YAML: {msg.data}")
        data = json.loads(msg.data)
        lvl = data.get("level_path", "")
        if lvl:
            self._load_level(lvl)
        else:
            self.level = None

    def _load_level(self, path:str):
        with open(path, 'r') as f:
            self.level = yaml.safe_load(f)
        self.start_t = time.time()
        self.get_logger().info(f"Loaded level {self.level['task_id']}")

    # def _ros_topic_exists(self, name: str) -> bool:
    #     topics = {t for (t, _types) in self.get_topic_names_and_types()}
    #     return name in topics

    # def _wait_for_gazebo(self, timeout=30.0) -> bool:
    #     """Wait until Gazebo is back (we use /model_states as the canary)."""
    #     t0 = time.time()
    #     while time.time() - t0 < timeout and rclpy.ok():
    #         if self._ros_topic_exists('/model_states'):
    #             return True
    #         rclpy.spin_once(self, timeout_sec=0.1)
    #     return False

    def _reader(self, topic: str):
        ignore = set(self.get_parameter('ignore_models').get_parameter_value().string_array_value
                    or self.get_parameter('ignore_models').value)

        while rclpy.ok():
            # 1) Wait for Gazebo to be up (handles initial start and hard resets)
            if not self._wait_for_gazebo(timeout=30.0):
                self.get_logger().warn("Gazebo not ready; retrying…")
                time.sleep(1.0)
                continue

            # 2) Start echoing contacts; suppress stderr noise
            try:
                proc = subprocess.Popen(
                    ['gz', 'topic', '-e', topic],
                    text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, bufsize=1
                )
                self._gz_proc = proc
            except FileNotFoundError:
                self.get_logger().error("`gz` CLI not found in PATH.")
                return
            
            self._last_contact_line_t = time.monotonic()
            self.get_logger().info(f"Referee: attached to {topic}")
            cur_c1 = None

            # 3) Read lines until Gazebo dies or this process exits
            for line in proc.stdout: 
                self._last_contact_line_t = time.monotonic() # heartbeat when ANY line arrives
                # self.get_logger().info(f"Contact line: {line.strip()}")
                m1, m2 = PAT_C1.search(line), PAT_C2.search(line)
                if m1:
                    cur_c1 = m1.group(1)
                if m2 and cur_c1:
                    c1, c2 = _model_of(cur_c1), _model_of(m2.group(1))
                    # self.get_logger().info(f"contact {c1} ↔ {c2}") # this works, but too verbose
                    if c1 not in ignore and c2 not in ignore:
                        with self._lock:
                            self._last_seen[_key(c1, c2)] = time.monotonic()
                    cur_c1 = None

            # 4) If we’re here, gz echo ended (Gazebo reset or quit) → clear cache and retry
            with self._lock:
                self._last_seen.clear()
            code = proc.wait()
            self.get_logger().warn(f"Lost contact stream (exit {code}); will reconnect…")
            time.sleep(0.5)

    def _ros_topic_exists(self, name: str) -> bool:
        return name in {t for (t, _types) in self.get_topic_names_and_types()}

    def _wait_for_gazebo(self, timeout=30.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout and rclpy.ok():
            # Gazebo (gzserver) publishes /clock when up
            if self._ros_topic_exists('/clock'):
                try:
                    # Confirm the contacts topic exists on Gazebo transport
                    out = subprocess.run(['gz','topic','-l'], capture_output=True, text=True, timeout=1.5)
                    if '/gazebo/default/physics/contacts' in out.stdout:
                        return True
                except Exception:
                    pass
            # rclpy.spin_once(self, timeout_sec=0.2)
        return False


    # def _reader(self, topic: str):
    #     proc = subprocess.Popen(['gz', 'topic', '-e', topic],
    #                             text=True, stdout=subprocess.PIPE, bufsize=1)
    #     cur_c1 = None
    #     ignore = set(self.get_parameter('ignore_models').get_parameter_value().string_array_value
    #                  or self.get_parameter('ignore_models').value)
    #     for line in proc.stdout:
    #         m1, m2 = PAT_C1.search(line), PAT_C2.search(line)
    #         if m1:
    #             cur_c1 = m1.group(1)
    #         if m2 and cur_c1:
    #             c1, c2 = _model_of(cur_c1), _model_of(m2.group(1))
    #             if c1 in ignore or c2 in ignore:
    #                 cur_c1 = None
    #                 continue
    #             with self._lock:
    #                 self._last_seen[_key(c1, c2)] = time.monotonic()
    #             cur_c1 = None

    #             # Debug output
    #             if {c1, c2}.isdisjoint({"ground_plane", "table_lightgray"}):
    #                 self.get_logger().debug(f"contact {c1} ↔ {c2}")

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
        t = getattr(self, "_last_contact_line_t", 0)
        # age = time.monotonic() - t
        # # self.get_logger().info(f"Referee tick at {time.time() - self.start_t:.2f}s, last contact line at {t:.2f}s ago")
        # self.get_logger().info(
        #     f"Referee tick at {time.time() - self.start_t:.2f}s, last contact line {age:.2f}s ago"
        # )
        # watchdog: restart stuck gz echo (happens after hard resets)
        if time.monotonic() - t > 2.0:
            p = getattr(self, "_gz_proc", None)
            if p and p.poll() is None:
                self.get_logger().warn("No contact stream lines recently; restarting gz echo…")
                try:
                    p.terminate()
                except Exception:
                    pass

        if not self.level:
            self.pub.publish(String(data=json.dumps({"status": "running", "reason": "no level loaded"})))
            return

        if (time.time() - self.start_t) > float(self.level.get("time_limit_s", 120)):
            self.pub.publish(String(data=json.dumps({
                "status": "timeout", "reason": "time limit exceeded"})))
            return
        
        cur = self._snapshot_pairs()
        succ = self.level["success"]
        fail = self.level.get("fail", {"collisions_true": [], "collisions_false": []})

        if cur != self._last_contacts:
            added = cur - self._last_contacts
            removed = self._last_contacts - cur
            self.get_logger().info(f"Contacts changed: added {added}, removed {removed}")
            self._last_contacts = cur

        # ----------   FAIL first   ----------
        for a, b in fail.get("collisions_true", []):
            if _key(a, b) in cur:
                self.pub.publish(String(data=json.dumps({
                    "status": "fail",
                    "reason": f"forbidden collision {a},{b}"})))
                return
        for a, b in fail.get("collisions_false", []):
            if _key(a, b) not in cur:
                self.pub.publish(String(data=json.dumps({
                    "status": "fail",
                    "reason": f"missing required separation {a},{b}"})))
                return

        # ----------   SUCCESS?   ----------
        ok_true  = all(_key(a, b) in cur  for a, b in succ.get("collisions_true",  []))
        ok_false = all(_key(a, b) not in cur for a, b in succ.get("collisions_false", []))
        
        # will maybe do full report (each pair:success/failure) later

        if ok_true and ok_false:
            self.pub.publish(String(data=json.dumps({
                "status": "success",
                "reason": "all success conditions satisfied"})))
        else:
            self.pub.publish(String(data=json.dumps({"status": "running"})))


def main():
    rclpy.init()
    n = Referee()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
