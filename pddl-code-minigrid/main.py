"""
Curriculum runner.

For every MiniGrid level in merged_curriculum2.json:
    1. Ask PlannerLLM for domain, problem, UP plan.
    2. Collect high-level action names from that plan.
    3. If any are missing in Agent, call CoderLLM once with:
         • the full missing-action set
         • each action's PDDL schema
         • the exact textual plan
         • the level name (for testing)
    4. Execute the finished plan; abort at first failure.
"""
from pathlib import Path
import json
import importlib
import datetime

import agent
from minigridenv import MiniGridEnv
from plannerLLM import PlannerLLM
from coderLLM import CoderLLM

CURRIC_FILE = Path(__file__).with_name("merged_curriculum2.json")

def reset():
    # If syntax errors occur in agent.py or agent_tmp.py, 
    # this should be run before importing Agent, or MiniGridEnv.
    import subprocess
    bash_command = """
    if ! diff -q ./maybe_not_useful_yet/agent_start-latest.py ./agent.py &>/dev/null; then
        cp ./maybe_not_useful_yet/agent_start-latest.py ./agent.py
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
    from plannerLLM import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, REFINEMENT_PROMPT_TEMPLATE
    from coderLLM import CODER_SYSTEM_PROMPT, CODER_INITIAL_TEMPLATE, CODER_FEEDBACK_TEMPLATE

    blocks = {
        "## PLANNER - system": SYSTEM_PROMPT,
        "## PLANNER - user template": USER_PROMPT_TEMPLATE,
        "## PLANNER - refinement template": REFINEMENT_PROMPT_TEMPLATE,
        "## CODER - system": CODER_SYSTEM_PROMPT,
        "## CODER - initial template": CODER_INITIAL_TEMPLATE,
        "## CODER - feedback template": CODER_FEEDBACK_TEMPLATE,
    }

    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"prompt_log_{datetime_str}.md"

    with log_file.open("w", encoding="utf-8") as f:
        for header, text in blocks.items():
            f.write(f"{header}\n\n{text.strip()}\n\n---\n\n")

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


def new_action_name(high_level_names, plan_str, schemas):

    import re
    def next_unused_name(name, existing_names):
        """
        If `name` is foo            → returns foo_v2, foo_v3, …              (first free)
        If `name` is foo_v3         → returns foo_v4, foo_v5, …              (first free)
        Works so long as every versioned name follows `<base>_v<digits>`.
        """
        m = re.match(r"^(.*?)(?:_v(\d+))?$", name)
        base   = m.group(1)
        suffix = m.group(2)

        i = int(suffix) + 1 if suffix else 2      # 2 if no suffix, else next int
        while f"{base}_v{i}" in existing_names:
            i += 1
        return f"{base}_v{i}"

    # compute new names
    name_map = {old: next_unused_name(old, high_level_names)
                for old in high_level_names}

    # set of new names only
    high_level_names_new = set(name_map.values())

    schemas_new = {name_map[old]: schemas[old].replace(f"(:action {old}",
                                                    f"(:action {name_map[old]}") for old in high_level_names}

    plan_str_new = plan_str
    for old, new in name_map.items():
        plan_str_new = plan_str_new.replace(f"{old}(", f"{new}(")

    return high_level_names_new, plan_str_new, schemas_new

def main():
    reset()
    reset_pddl()
    prompt_log()

    curriculum = json.loads(CURRIC_FILE.read_text())
    planner = PlannerLLM()
    coder   = CoderLLM()

    for cat in curriculum:
        reset_pddl()  # reset PDDL files for each category
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
                print(f"Plan: {plan_str}")
                high_level_names = {s.action.name.replace("-", "_")
                                    for s in up_result.plan.actions}
                

                # ---------- 2 · IMPLEMENT MISSING ACTIONS ---------------
                print("🔍 checking Agent for missing actions...")
                missing = high_level_names - set(dir(agent.Agent))
                if missing:
                    schemas = planner.get_action_schemas(missing)
                    coder.implement_actions(
                        actions=missing,
                        pddl_schemas=schemas,
                        plan_str=plan_str,
                        test_env=env_name,
                    )
                    importlib.reload(agent)  # hot-reload new methods

                else:

                    # ---------- 3 · RUN FULL PLAN ---------------------------
                    print("🚀 running full plan...")
                    env = MiniGridEnv(env_name)
                    outcome = env.run_sim(plan_str)
                    print(f"Plan execution result: {outcome}")
                    env.end_env()

                    # ---------- 4 · EDIT AGENT IF NEEDED ------------------------
                    if outcome != "success":
                        print(f"❌ level {env_name}")# failed: {outcome}")

                        # Implement v2 of functions, and append code (not replace).
                        print("🔄 implementing v2 actions...")
                        schemas = planner.get_action_schemas(high_level_names)

                        high_level_names_new, plan_str_new, schemas_new = new_action_name(
                            high_level_names, plan_str, schemas)
                        
                        # print(f"New all: {high_level_names_new};; {plan_str_new};; {schemas_new}")
                        
                        coder.implement_actions(
                            actions=high_level_names_new,
                            pddl_schemas=schemas_new,
                            plan_str=plan_str_new,
                            test_env=env_name,
                        )
                        importlib.reload(agent)
                    
                    # else:
                    #     print(f"✅ level {env_name} solved with actions: {', '.join(high_level_names)}")


    print("\n🏁  All curriculum levels solved.")


if __name__ == "__main__":
    main()
