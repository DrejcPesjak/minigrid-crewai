"""
Curriculum runner.

For every MiniGrid level in merged_curriculum2.json:
    1. Ask PlannerLLM for domain, problem, UP plan.
    2. Collect high-level action names from that plan.
    3. If any are missing in Agent, call CoderLLM once with:
         • the full missing-action set
         • each action’s PDDL schema
         • the exact textual plan
         • the level name (for testing)
    4. Execute the finished plan; abort at first failure.
"""
from pathlib import Path
import json
import importlib

from agent import Agent
from minigridenv import MiniGridEnv
from plannerLLM import PlannerLLM
from coderLLM import CoderLLM

CURRIC_FILE = Path(__file__).with_name("merged_curriculum2.json")

def reset():
    import subprocess
    bash_command = """
    if ! diff -q ./maybe_not_useful_yet/agent_start-latest.py ./agent.py &>/dev/null; then
        cp ./maybe_not_useful_yet/agent_start-latest.py ./agent.py
        cp ./maybe_not_useful_yet/agent_start-latest.py ./agent_tmp.py
    fi
    rm -f ./domain.pddl ./problem.pddl
    """
    subprocess.run(bash_command, shell=True, check=True)

def plan_to_string(up_plan) -> str:
    """Return MiniGridEnv-ready string '[foo(), bar(obj)]'."""
    parts = []
    for step in up_plan.actions:
        name = step.action.name.replace("-", "_")
        if step.actual_parameters:
            params = ", ".join(map(str, step.actual_parameters))
            parts.append(f"{name}({params})")
        else:
            parts.append(f"{name}()")
    return "[" + ", ".join(parts) + "]"


def main():
    reset()
    curriculum = json.loads(CURRIC_FILE.read_text())
    planner = PlannerLLM()
    coder   = CoderLLM()

    for cat in curriculum:
        for lvl in cat["levels"]:
            for env_name in lvl["configs"]:
                print(f"\n=== {env_name} ===")

                # ---------- 1 · PLAN ------------------------------------
                print("📝 planning...")
                _, up_result, _ = planner.plan(
                    meta={
                        "category_name": cat["category_name"],
                        "skill": cat["skill"],
                        "level_name": lvl["name"],
                        "level_description": lvl["description"],
                        "env_name": env_name,
                    }
                )
                plan_str = plan_to_string(up_result.plan)
                high_level_names = {s.action.name.replace("-", "_")
                                    for s in up_result.plan.actions}

                # ---------- 2 · IMPLEMENT MISSING ACTIONS ---------------
                print("🔍 checking Agent for missing actions...")
                missing = high_level_names - set(dir(Agent))
                if missing:
                    schemas = planner.get_action_schemas(missing)
                    coder.implement_actions(
                        actions=missing,
                        pddl_schemas=schemas,
                        plan_str=plan_str,
                        test_env=env_name,
                    )
                    importlib.reload(Agent)  # hot-reload new methods

                # ---------- 3 · RUN FULL PLAN ---------------------------
                print("🚀 running full plan...")
                env = MiniGridEnv(env_name)
                outcome = env.run_sim(plan_str)
                env.end_env()
                if outcome != "success":
                    raise RuntimeError(f"❌ level {env_name} failed: {outcome}")
                print("✓ solved")

    print("\n🏁  All curriculum levels solved.")


if __name__ == "__main__":
    main()
