"""
main.py – Sense → Plan → Code → Act loop over the BabyAI / MiniGrid curriculum.

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

import shutil
import importlib, json, re, subprocess, traceback
from pathlib import Path
from typing import Set, Tuple

import agent_tmp
from coderLLM   import CoderLLM
from minigridenv import MiniGridEnv, Outcome
from plannerLLM import PlannerLLM, PlanBundle

# ───────────── constants ─────────────────────────────────────────────
CURRIC_FILE          = Path(__file__).with_name("merged_curriculum2.json")
MAX_SPA_ROUNDS       = 5          # sense-plan-code-act iterations / cfg
MAX_SEMANTIC_RETRY   = 5          # coder retries after semantic error
PDDL_FILES           = ("domain.pddl", "problem.pddl")
AGENT_FILE           = Path(__file__).with_name("agent.py")
TMP_FILE             = Path(__file__).with_name("agent_tmp.py")

# ───────────── helpers ------------------------------------------------
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

def end_logging():
    """ Save agent_tmp.py and domain/problem.pddl files to ./new_logs/logs_<timestamp>/ """
    import datetime, shutil
    from pathlib import Path

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("new_logs") / f"logs_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(TMP_FILE, log_dir / "agent_tmp.py")
    for f in PDDL_FILES:
        if Path(f).exists():
            shutil.copy(f, log_dir / f)
    print(f"🔚 logs saved to {log_dir}")

# ───────────── main ---------------------------------------------------
def main():
    reset_agent()
    curriculum = json.loads(Path(CURRIC_FILE).read_text())

    planner = PlannerLLM()   # handles big→small swap inside itself
    coder   = CoderLLM()

    for cat in curriculum:
        print(f"\n===== CATEGORY  {cat['category_name']} =====")
        reset_pddl()                 # fresh per category
        pddl_cache = {"pddl": None, "trusted": False} # cache the *latest* PDDL plus a flag if it has already solved a level

        for lvl in cat["levels"]:
            for cfg in lvl["configs"]:
                print(f"\n--- {cfg} ---")
                env   = MiniGridEnv(cfg, seed=42)
                meta={
                        "category_name": cat["category_name"],
                        "skill": cat["skill"],
                        "level_name": lvl["name"],
                        "level_description": lvl["description"],
                        "env_name": cfg,
                    }
                
                plan_failed = False  # per-config flag
                for spa in range(1, MAX_SPA_ROUNDS+1):
                    print(f"\n[S-P-C-A] round {spa}")

                    # SENSE ──────────────────────────────
                    snap = env.snapshot()

                    # PLAN  (planner has its own 0-N loop) ─
                    bundle:PlanBundle = planner.plan(
                        snap, 
                        meta, 
                        pddl_hint    = pddl_cache["pddl"],
                        pddl_trusted = pddl_cache["trusted"],
                        plan_failed  = plan_failed,
                    )
                    pddl_cache["pddl"] = (bundle.domain, bundle.problem)
                    print(f"📜 plan: {bundle.plan_str}")
                    names = hl_names(bundle.plan_str)
                    missing = names - set(dir(agent_tmp.Agent))

                    semantic_retry = 0
                    out:Outcome = None
                    while True:
                        checkpoint = env._checkpoint.copy()
                        
                        # CODE  (coder handles syntax / merge) 
                        if missing:
                            result = coder.implement_actions(
                                actions       = missing,
                                pddl_schemas  = {n: bundle.action_schemas.get(n,"") for n in missing},
                                plan_str      = bundle.plan_str,
                                agent_state   = env._agent_state() if not out else out.agent_state,
                                past_error_log= None if not out else out.msg + "\n" + out.trace,
                            )
                            if result.status != "ok":
                                print(f"❌ coder failed at syntax stage ({result.status})")
                                plan_failed = True
                                break  # skip SPA loop
                            importlib.reload(agent_tmp)
                            env.replay_checkpoint()  # resets the agent
                        else:
                            print("✅ no missing methods, skipping coder round") # just for debug

                        # ACT ─────────────────────────────
                        out:Outcome = env.run_sim(bundle.plan_str)
                        print(f"{out.status}: {out.msg}")

                        if out.status == "success":
                            print("✅ level solved")
                            shutil.copy(TMP_FILE, AGENT_FILE)
                            pddl_cache["trusted"] = True
                            break  # out of while
                        if out.status == "goal_not_reached":
                            missing = names

                        if out.status in ("missing_method", "syntax_error",
                                          "reward_failed", "stuck"):
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

                        print(f"🔄 retrying coder round {semantic_retry}...")

                    if out.status == "success":
                        break  # next config
                    # else: fall back to SPA round → new plan

                env.end_env()

    print("\n🏁 curriculum finished")

# ───────────── entry point --------------------------------------------
if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt:
        print("\n⛔ interrupted")
    except Exception:
        traceback.print_exc()
    finally:
        end_logging()
