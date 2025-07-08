"""
main.py – Sense → Plan → Code → Act loop over the BabyAI / MiniGrid curriculum.

Workflow per **environment config**

    env  = MiniGridEnv(cfg)
    prev_pddl = None
    for spa_round ≤ MAX_SPA:
        snapshot = env.snapshot()                   # SENSE
        bundle   = planner.plan(snapshot, meta, prev_pddl)     # PLAN
        prev_pddl = (bundle.domain, bundle.problem)

        ensure_agent_code(bundle)                   # CODE  (may call coder)
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

import datetime
import importlib
import json
import re
import traceback
from pathlib import Path
from typing import Dict, Set, Tuple

import agent                                # starter skeleton
from coderLLM import CoderLLM, CoderResult
from minigridenv import MiniGridEnv, Outcome
from plannerLLM import PlannerLLM, PlanBundle

# --------------------------------------------------------------------------- #
#  CONSTANTS / FILES
# --------------------------------------------------------------------------- #

CURRIC_FILE  = Path(__file__).with_name("merged_curriculum2.json")
MAX_SPA      = 10            # sense-plan-code-act iterations per config
MAX_CODER_RE = 5             # extra coder retries after runtime errors

# --------------------------------------------------------------------------- #
#  UTILITIES  (unchanged helpers from old script) 
# --------------------------------------------------------------------------- #

def reset():
    # If syntax errors occur in agent.py or agent_tmp.py, 
    # this should be run before importing Agent, or MiniGridEnv.
    import subprocess
    bash_command = """
    if ! diff -q ./agent_start-latest.py ./agent.py &>/dev/null; then
        cp ./agent_start-latest.py ./agent.py
    fi
    cp ./agent.py ./agent_tmp.py
    
    #rm -f ./domain.pddl ./problem.pddl
    """
    subprocess.run(bash_command, shell=True, check=True)
    importlib.reload(agent)  # reload Agent class
    print("Reset agent.py and agent_tmp.py to their initial state.")

def reset_pddl():
    """Reset PDDL files to their initial state."""
    import subprocess
    bash_command = """
    rm -f ./domain.pddl ./problem.pddl
    """
    subprocess.run(bash_command, shell=True, check=True)
    print("Reset domain.pddl and problem.pddl to their initial state.")

def prompt_log():
    from plannerLLM import SYSTEM_PROMPT as P_SYS, USER_PROMPT_TEMPLATE_FIRST as P_USR
    from coderLLM   import CODER_SYSTEM_PROMPT as C_SYS, CODER_INITIAL_TEMPLATE as C_USR
    txt = "\n\n---\n\n".join([
        "## Planner system", P_SYS,
        "## Planner user",   P_USR,
        "## Coder system",   C_SYS,
        "## Coder user",     C_USR,
    ])
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    (Path(__file__).parent / "logs").mkdir(exist_ok=True)
    (Path(__file__).parent / "logs" / f"prompts_{ts}.md").write_text(txt)

def end_log():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(__file__).parent / "new_logs" / ts
    out.mkdir(parents=True, exist_ok=True)
    for f in ("domain.pddl","problem.pddl","agent_tmp.py"):
        p = Path(f);  p.exists() and (out / f).write_text(p.read_text())
    print(f"Logs saved → {out}")

# --------------------------------------------------------------------------- #
#  SMALL HELPERS
# --------------------------------------------------------------------------- #

def extract_hl_names(plan_str:str) -> Set[str]:
    return {m.group(1).replace("-","_")
            for m in re.finditer(r'([a-zA-Z][\w-]*)\s*\(', plan_str)}

def ensure_agent_code(bundle: PlanBundle,
                      coder : CoderLLM,
                      agent_state: dict,
                      error_log: str | None = None) -> None:
    """
    Implement or patch high-level actions until import succeeds.
    Raises RuntimeError if coder exhausts retries.
    """
    hl_names = extract_hl_names(bundle.plan_str)
    missing  = hl_names - set(dir(agent.Agent))
    # even if none missing we may still call coder in case of semantic patch
    if not missing and not error_log:
        return

    tries = 0
    while True:
        result: CoderResult = coder.implement_actions(
            actions       = hl_names if error_log else missing,
            pddl_schemas  = {n: bundle.action_schemas.get(n,"")
                             for n in hl_names},
            plan_str      = bundle.plan_str,
            agent_state   = agent_state,
            past_error_log= error_log,
        )
        if result.status == "ok":
            importlib.reload(agent)
            return
        if result.status in ("merge_error","reload_error"):
            tries += 1
            error_log = result.trace
            if tries >= MAX_CODER_RE:
                raise RuntimeError(f"Coder stuck after {tries} retries")
            continue
        raise RuntimeError("Coder exhausted retries")

# def ensure_agent_code(bundle, coder, agent_state, error_log=None):
#     hl = extract_hl_names(bundle.plan_str)
#     missing = hl - set(dir(agent.Agent))
#     if not missing and not error_log:
#         return
#     res = coder.implement_actions(
#         actions       = hl if error_log else missing,
#         pddl_schemas  = {n: bundle.action_schemas.get(n, "") for n in hl},
#         plan_str      = bundle.plan_str,
#         agent_state   = agent_state,
#         past_error_log= error_log,
#     )
#     if res.status != "ok":
#         raise RuntimeError(f"Coder failed: {res.status}")
#     importlib.reload(agent)


# --------------------------------------------------------------------------- #
#  MAIN CURRICULUM LOOP
# --------------------------------------------------------------------------- #

def main():
    prompt_log()
    curriculum = json.loads(CURRIC_FILE.read_text())

    planner = PlannerLLM()
    coder   = CoderLLM()

    checkpoint_cat   = None            # set string to resume mid-curriculum
    checkpoint_seen  = checkpoint_cat is None

    if checkpoint_cat is None:
        reset()

    for cat in curriculum:
        if not checkpoint_seen:
            if cat["category_name"] == checkpoint_cat:
                checkpoint_seen = True
            else:
                print(f"⏩ skipping category {cat['category_name']}")
                continue
        
        reset_pddl()  # reset PDDL files for each category

        for lvl in cat["levels"]:
            for env_name in lvl["configs"]:
                print(f"\n=== {env_name} ===")

                # ---------- initialise env & meta -----------------------
                env  = MiniGridEnv(env_name, seed=42)
                meta = {
                    "category_name"   : cat["category_name"],
                    "skill"           : cat["skill"],
                    "level_name"      : lvl["name"],
                    "level_description": lvl["description"],
                    "env_name"        : env_name,
                }
                prev_pddl: Tuple[str,str] | None = None

                # ---------- SPA loop ------------------------------------
                for spa_round in range(1, MAX_SPA + 1):
                    print(f"\n[S-P-C-A] round {spa_round}")

                    # ---------- SENSE -----------------------------
                    snapshot = env.snapshot()

                    # ---------- PLAN ------------------------------
                    bundle   : PlanBundle = planner.plan(     
                        snapshot, meta, prev_pddl)
                    prev_pddl = (bundle.domain, bundle.problem)

                    # ---------- CODE -----------------------------
                    try:                                      
                        ensure_agent_code(bundle,
                                           coder,
                                           agent_state=snapshot,
                                           error_log=None)
                    except RuntimeError as e:
                        print(f"Coder unrecoverable: {e}")
                        break

                    # ---------- ACT ------------------------------
                    outcome : Outcome = env.run_sim(bundle.plan_str) 
                    print(f"→ {outcome.status}: {outcome.msg}")


                    # ---------- HANDLE OUTCOME -------------------
                    if outcome.status == "success":
                        print(f"✅ solved {env_name}")
                        break

                    if outcome.status in ("missing_method","syntax_error"):
                        # patch same HL names using error log, then retry plan
                        try:
                            ensure_agent_code(bundle,
                                              coder,
                                              agent_state=outcome.agent_state,
                                              error_log=outcome.msg + "\n" + outcome.trace)
                            env.replay_checkpoint()
                            continue          # stay in same SPA round count
                        except RuntimeError as e:
                            print(f"Coder failed repairing syntax: {e}")
                            break

                    if outcome.status in ("stuck","goal_not_reached","reward_failed"):
                        # plan or code semantics wrong – get new snapshot+plan
                        env.replay_checkpoint()
                        continue

                    # runtime_error or unknown → abort this config
                    print(f"Fatal in {env_name}: {outcome.msg}")
                    break   # exit SPA loop for this config

                # ensure env closed before next config
                env.end_env()

    print("\n🏁 Curriculum loop finished.")

# --------------------------------------------------------------------------- #
#  ENTRY-POINT
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user.")
    except Exception as e:
        traceback.print_exc()
        print(f"❗ Fatal: {e}")
    finally:
        end_log()
