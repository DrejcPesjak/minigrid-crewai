"""
main.py : Sense → Plan → Code → Act loop over the BabyAI / MiniGrid curriculum.

Workflow per **environment config**

    env  = MiniGridEnv(cfg)
    prev_pddl = None
    for spa_round ≤ MAX_SPA:
        snapshot = env.snapshot()                   # SENSE
        bundle   = planner.plan(snapshot, meta, prev_pddl)     # PLAN
        prev_pddl = (bundle.domain, bundle.problem)

        for semantic_round ≤ MAX_CODER_RE:
            result = coder.implement_actions(bundle)       # CODE 
            outcome  = env.run_sim(bundle.plan_str)     # ACT

            handle_outcome(outcome)  →  either
                • success   → next config
                • syntax/impl error → coder patch, env.replay_checkpoint(), repeat
                • plan inadequate   → continue SPA loop → new plan
                • fatal             → abort curriculum

The outer curriculum / checkpoint logic (skip categories until some
check-point) is kept from the original script.
"""

from __future__ import annotations

import shutil, datetime
import importlib, json, re, subprocess, traceback, argparse
from pathlib import Path
from typing import Set, Tuple

import agent_tmp
from coderLLM   import CoderLLM
from minigridenv import MiniGridEnv, Outcome
from plannerLLM import PlannerLLM, PlanBundle
from metrics_logger import MetricsLogger

# ───────────── constants ─────────────────────────────────────────────
CURRIC_FILE          = Path(__file__).with_name("merged_curriculum2.json")
MAX_SPA_ROUNDS       = 5          # sense-plan-code-act iterations / cfg
MAX_SEMANTIC_RETRY   = 5          # coder retries after semantic error
PDDL_FILES           = ("domain.pddl", "problem.pddl")
AGENT_FILE           = Path(__file__).with_name("agent.py")
TMP_FILE             = Path(__file__).with_name("agent_tmp.py")

# ───────────── globals ─────────────────────────────────────────────
log_dir: Path | None = None
agent_version: int = 0

# ───────────── helpers ────────────────────────────────────────────
ACT_RE = re.compile(r'([A-Za-z][\w-]*)\s*\(')
def hl_names(plan:str) -> Set[str]:
    return {m.group(1).replace("-", "_") for m in ACT_RE.finditer(plan)}

def reset_agent():
    subprocess.run("cp agent_start-latest.py agent.py && cp agent.py agent_tmp.py",
                   shell=True, check=True)
    importlib.reload(agent_tmp)
    print("🔄 agent reset")

def reset_pddl():
    for f in PDDL_FILES:
        Path(f).unlink(missing_ok=True)
    print("🔄 pddl reset")


def prompt_log(dir: Path = Path(".")):
    from plannerLLM import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE_FIRST, USER_PROMPT_TEMPLATE_PDDL, USER_PROMPT_TEMPLATE_REPAIR
    from coderLLM import CODER_SYSTEM_PROMPT, CODER_INITIAL_TEMPLATE, CODER_FEEDBACK_TEMPLATE

    blocks = {
        "## PLANNER - system": SYSTEM_PROMPT,
        "## PLANNER - user template first": USER_PROMPT_TEMPLATE_FIRST,
        "## PLANNER - user template pddl": USER_PROMPT_TEMPLATE_PDDL,
        "## PLANNER - refinement template": USER_PROMPT_TEMPLATE_REPAIR,
        "## CODER - system": CODER_SYSTEM_PROMPT,
        "## CODER - initial template": CODER_INITIAL_TEMPLATE,
        "## CODER - feedback template": CODER_FEEDBACK_TEMPLATE,
    }
    
    log_file = dir / "prompt_log.md"

    with log_file.open("w", encoding="utf-8") as f:
        for header, text in blocks.items():
            f.write(f"{header}\n\n{text.strip()}\n\n---\n\n")


def end_logging(fail:bool=False):
    """ Save agent_tmp.py and domain/problem.pddl files to ./new_logs/logs_<timestamp>/ """
    global log_dir, agent_version

    if log_dir is None:
        # create new log directory with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("new_logs") / f"logs_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)

        # save prompt log (only once)
        prompt_log(log_dir)

    prefix = "fail_" if fail else ""
    # agent snapshot
    shutil.copy2(TMP_FILE, log_dir / f"{prefix}agent_v{agent_version}.py")

    # PDDL snapshots
    for name in PDDL_FILES:
        src = Path(name)
        if src.exists():
            dst = log_dir / f"{prefix}{src.stem}_v{agent_version}{src.suffix}"
            shutil.copy2(src, dst)
    
    print(f"🔚 logs saved (v{agent_version}) → {log_dir}")
    agent_version += 1

def _load_checkpoint_pddl() -> Tuple[str, str] | None:
    """Return (domain_text, problem_text) if both PDDL files exist, else None."""
    dom, prob = (Path(n) for n in PDDL_FILES)
    if dom.exists() and prob.exists():
        return dom.read_text(), prob.read_text()
    return None

# ───────────── main loop ────────────────────────────────────────────────
def main(start_category: str | None = None,
         keep_agent: bool = False,
         keep_pddl: bool = False):
    global log_dir
    
    # optionally preserve current agent state
    if not keep_agent:
        reset_agent()
    else:
        # ensure TMP_FILE exists so logging doesn't explode later
        if not TMP_FILE.exists() and AGENT_FILE.exists():
            shutil.copy2(AGENT_FILE, TMP_FILE)
        print("⏭ skipping agent reset (keep_agent=True)")

    curriculum = json.loads(Path(CURRIC_FILE).read_text())

    if log_dir is None:
        log_dir = Path("new_logs") / f"logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir.mkdir(parents=True, exist_ok=True)

    metrics = MetricsLogger(log_dir)
    metrics.register_signal_handlers()

    planner = PlannerLLM(metrics=metrics)   # handles big→small swap inside itself
    coder   = CoderLLM(metrics=metrics)

    # start/resume logic
    skipping = False
    tgt = None
    if start_category:
        tgt = start_category.lower()
        skipping = True
        print(f"⏩ will skip categories until '{start_category}'")
        if not any(cat["category_name"].lower() == tgt for cat in curriculum):
            print(f"⚠️ start_category '{start_category}' not found; running from beginning.")
            skipping = False

    # preload PDDL if requested
    resume_pddl = _load_checkpoint_pddl() if keep_pddl else None

    for cat in curriculum:
        is_resume_cat = False
        if skipping:
            if cat["category_name"].lower() != tgt:
                print(f"⏩ skipping category {cat['category_name']}")
                continue
            skipping = False
            is_resume_cat = True

        print(f"\n===== CATEGORY  {cat['category_name']} =====")

        if is_resume_cat and keep_pddl:
            if resume_pddl:
                print("⏭ skipping PDDL reset & seeding from checkpoint (keep_pddl=True)")
                pddl_cache = {"pddl": resume_pddl, "trusted": True}
            else:
                print("⚠️ keep_pddl requested but no PDDL files found; doing fresh reset.")
                reset_pddl()
                pddl_cache = {"pddl": None, "trusted": False}
        else:
            reset_pddl()  # fresh per (non-resume) category
            pddl_cache = {"pddl": None, "trusted": False} # latest PDDL + trust flag

        # pddl_cache  # cache the *latest* PDDL plus a flag if it has already solved a level

        for lvl in cat["levels"]:
            for cfg in lvl["configs"]:
                print(f"\n--- {cfg} ---")
                seed = 1
                env   = MiniGridEnv(cfg, seed=seed)
                meta={
                        "category_name": cat["category_name"],
                        "skill": cat["skill"],
                        "level_name": lvl["name"],
                        "level_description": lvl["description"],
                        "env_name": cfg,
                    }
                if metrics:
                    metrics.start_level(level_name=meta.get("env_name","unknown"),
                                        lvl_group=meta.get("level_name","unknown"),
                                        category=meta.get("category_name","unknown"),
                                        seed=seed)
                plan_failed = False  # per-config flag
                for spa in range(1, MAX_SPA_ROUNDS+1):
                    print(f"\n[S-P-C-A] round {spa}")

                    # SENSE ──────────────────────────────
                    snap = env.snapshot()

                    if metrics:
                        metrics.inc_spca_round()

                    # PLAN  (planner has its own 0-N loop) ─
                    bundle:PlanBundle = planner.plan(
                        snap, 
                        meta, 
                        pddl_hint    = pddl_cache["pddl"],
                        pddl_trusted = pddl_cache["trusted"],
                        plan_failed  = plan_failed,
                    )
                    pddl_cache["pddl"] = (bundle.domain, bundle.problem)
                    pddl_cache["trusted"] = False
                    print(f"📜 plan: {bundle.plan_str}")
                    names = hl_names(bundle.plan_str)
                    missing = names - set(dir(agent_tmp.Agent))

                    semantic_retry = 0
                    out:Outcome = None
                    checkpoint = env._checkpoint.copy()
                    while True:
                        # CODE  (coder handles syntax / merge) 
                        if missing:
                            if metrics and semantic_retry == 0:
                                metrics.inc_coder_first()
                            elif metrics and semantic_retry > 0:
                                metrics.inc_coder_semantic()

                            result = coder.implement_actions(
                                actions       = missing,
                                pddl_schemas  = {n: bundle.action_schemas.get(n,"") for n in missing},
                                plan_str      = bundle.plan_str,
                                agent_state   = env._agent_state() if not out else out.agent_state,
                                past_error_log= None if not out else out.msg + "\n" + out.trace,
                            )
                            if result.status != "ok":
                                print(f"❌ coder failed at syntax stage ({result.status})")
                                shutil.copy(AGENT_FILE, TMP_FILE)  # reset to last working agent
                                importlib.reload(agent_tmp)
                                plan_failed = True
                                break  # skip SPA loop
                            
                            importlib.reload(agent_tmp)
                            env.reload_agent()  # reload agent with new code
                        else:
                            print("✅ no missing methods, skipping coder round") # just for debug

                        # ACT ─────────────────────────────
                        if metrics:
                            metrics.inc_simulation_run()
                        out:Outcome = env.run_sim(bundle.plan_str)
                        print(f"{out.status}: {out.msg}")

                        if out.status == "success":
                            print("✅ level solved")
                            shutil.copy(TMP_FILE, AGENT_FILE)
                            pddl_cache["trusted"] = True
                            if metrics:
                                metrics.end_level(outcome=out.status)
                            end_logging() # save logs
                            break  # next config

                        if out.status == "goal_not_reached":
                            missing = names

                        if out.status in ("missing_method", "syntax_error",
                                          "reward_failed", "stuck", 
                                          "runtime_error"):
                            # coder retry, rollback to checkpoint
                            missing = names # re-implment all HL actions
                            # maybe do v2 here

                        # handle "runtime_error"

                        # semantic error → patch same code
                        semantic_retry += 1
                        if semantic_retry > MAX_SEMANTIC_RETRY:
                            print("⚠️  semantic retries exhausted"); 
                            plan_failed = True
                            break

                        env._checkpoint = checkpoint
                        env.replay_checkpoint()

                        print(f"🔄 retrying semantic coder round {semantic_retry}...")

                    if out:
                        if out.status == "success":
                            break  # next config
                        elif spa == MAX_SPA_ROUNDS:
                            print("⚠️  SPA retries exhausted"); 
                            if metrics:
                                metrics.end_level(outcome="fail")
                    
                    # else: fall back to SPA round → new plan

                env.end_env()

    print("\n🏁 curriculum finished")

# ───────────── entry point --------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-category", type=str, default=None,
                        help="Category name to resume from (case-insensitive).")
    parser.add_argument("--keep-agent", action="store_true",
                        help="Skip reset_agent() and continue with current agent code.")
    parser.add_argument("--keep-pddl", action="store_true",
                        help="Skip reset_pddl() for the start category and seed pddl_cache from existing domain/problem files.")
    args = parser.parse_args()

    #python main.py --start-category static_obstacle_navigation --keep-agent --keep-pddl

    ok = False
    try: 
        main(start_category=args.start_category,
            keep_agent=args.keep_agent,
            keep_pddl=args.keep_pddl)
        ok = True
    except KeyboardInterrupt:
        print("\n⛔ interrupted")
        traceback.print_exc()
    except Exception:
        traceback.print_exc()
    finally:
        end_logging(fail=not ok)

