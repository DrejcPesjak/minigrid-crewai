# metrics_logger.py
import json, threading, time, uuid, signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

class MetricsLogger:
    """
    Per-level metrics with tiny, explicit counters.
    - Call start_level(...) once per level (right after you build `meta`)
    - Call inc_spca_round() where your "SPCA round" currently happens
    - Call inc_simulation_run() right before env.run_sim(...)
    - Coder: main() calls inc_coder_first()/inc_coder_semantic(); CoderLLM uses log_coder_syntax_repair(...)
    - Planner: PlannerLLM calls log_planner_mode(kind=...), which increments per-mode & logs the call
    - Always call end_level(outcome=...) on success OR failure. For Ctrl-C, use register_signal_handlers(...)
    """

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / "metrics.jsonl"
        self._lock = threading.Lock()
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]

        # per-level state
        self._level_ctx: Optional[Dict[str, Any]] = None
        self._level_counters: Dict[str, int] = {}
        self._level_aggregates: Dict[str, Any] = {}
        self._level_start_ts: Optional[float] = None
        self._level_open: bool = False  # prevents double end_level()

    # ---------- internals ----------
    def _write(self, obj: dict):
        obj = dict(obj)
        obj.setdefault("run_id", self.run_id)
        obj.setdefault("timestamp", datetime.now(timezone.utc).isoformat() + "Z")
        if self._level_ctx:
            obj.setdefault("level_context", dict(self._level_ctx))
        line = json.dumps(obj, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n"); f.flush()

    def _inc(self, name: str, by: int = 1):
        self._level_counters[name] = self._level_counters.get(name, 0) + by
        self._write({"type": "counter_inc", "name": name, "value": self._level_counters[name]})

    # ---------- lifecycle ----------
    def start_level(self, level_name: str, lvl_group: str, category: str, seed: int | None = None, extra: dict | None = None):
        self._level_ctx = {"level_name": level_name, "lvl_group": lvl_group, "category": category}
        if seed is not None: self._level_ctx["seed"] = seed
        if extra: self._level_ctx.update(extra)
        self._level_counters = {
            # totals
            "spca_rounds": 0,
            "simulation_runs": 0,
            # planner
            "planner_calls": 0,
            "planner_fresh": 0,
            "planner_reuse": 0,
            "planner_replan": 0,
            "planner_syntax": 0,
            # coder
            "coder_calls": 0,
            "coder_first": 0,
            "coder_semantic": 0,
            "coder_syntax": 0,
        }
        self._level_aggregates = {
            "planner_call_durations_s": [],
            "coder_call_durations_s": [],
        }
        self._level_start_ts = time.time()
        self._level_open = True
        self._write({"type": "level_start", "level": self._level_ctx})

    def end_level(self, outcome: str, extra_summary: dict | None = None):
        # idempotent: safe to call multiple times
        if not self._level_open: return
        self._level_open = False

        elapsed_min = None
        if self._level_start_ts is not None:
            elapsed_min = (time.time() - self._level_start_ts) / 60.0

        summary = {
            "type": "level_summary",
            "level": self._level_ctx,
            "outcome": outcome,  # "success" | "fail" | "aborted"
            "elapsed_min": elapsed_min,
            "counters": dict(self._level_counters),
            "aggregates": {
                "planner_calls_count": len(self._level_aggregates["planner_call_durations_s"]),
                "planner_call_duration_mean_s":
                    (sum(self._level_aggregates["planner_call_durations_s"]) / max(1, len(self._level_aggregates["planner_call_durations_s"])))
                    if self._level_aggregates["planner_call_durations_s"] else None,
                "coder_calls_count": len(self._level_aggregates["coder_call_durations_s"]),
                "coder_call_duration_mean_s":
                    (sum(self._level_aggregates["coder_call_durations_s"]) / max(1, len(self._level_aggregates["coder_call_durations_s"])))
                    if self._level_aggregates["coder_call_durations_s"] else None,
            }
        }
        if extra_summary: summary.update(extra_summary)

        # append to jsonl
        self._write(summary)

        # write a per-level JSON snapshot
        lvl = self._level_ctx or {}
        safe = f"{lvl.get('category','unknown')}_{lvl.get('lvl_group','unknown').replace(' ', '')}_{lvl.get('level_name','unknown')}"
        with open(self.out_dir / f"summary_{safe}_{self.run_id}.json", "w", encoding="utf-8") as f:
            json.dump({"run_id": self.run_id, "summary": summary}, f, indent=2, default=str)

        # reset
        self._level_ctx = None
        self._level_counters = {}
        self._level_aggregates = {}
        self._level_start_ts = None

    # ---------- counters you asked for ----------
    # spca rounds (call where your current "SPCA round" marker is)
    def inc_spca_round(self): self._inc("spca_rounds")

    # simulation run (call immediately before env.run_sim(...))
    def inc_simulation_run(self): self._inc("simulation_runs")

    # coder buckets (main increments first & semantic; coder increments syntax internally)
    def inc_coder_first(self): self._inc("coder_first"); self._inc("coder_calls")
    def inc_coder_semantic(self): self._inc("coder_semantic"); self._inc("coder_calls")

    def log_coder_call_duration(self, duration_s: float):
        self._level_aggregates["coder_call_durations_s"].append(duration_s)
        self._write({"type": "coder_call", "duration_s": duration_s})

    def log_coder_syntax_repair(self, repair_step: int, error_msg: str, patched_code: str | None = None):
        self._inc("coder_syntax"); self._inc("coder_calls")
        self._write({"type": "coder_syntax_repair", "repair_step": repair_step, "error_msg": error_msg, "patched_code": patched_code})

    # planner buckets (single call that logs + increments per mode)
    def log_planner_mode(self, kind: str, model: str, duration_s: float):
        # kind ∈ {"fresh","reuse","replan","syntax"}
        self._inc("planner_calls")
        k = kind.lower()
        if k in ("fresh", "reuse", "replan", "syntax"):
            self._inc(f"planner_{k}")
        self._level_aggregates["planner_call_durations_s"].append(duration_s)
        self._write({"type": "planner_call", "kind": kind, "model": model, "duration_s": duration_s})

    # ---------- Ctrl-C / SIGTERM safety ----------
    def register_signal_handlers(self):
        def _handler(signum, frame):
            try:
                # write an aborted summary if a level is open
                if self._level_open:
                    self.end_level(outcome="aborted")
            finally:
                raise KeyboardInterrupt  # let your code stop as usual
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
