# Load the existing curriculum and environment metadata files
import json
from pathlib import Path


with open("micro_task_curriculum2.json", "r") as f:
    curriculum = json.load(f)

with open("minigrid_babyai_envs.json", "r") as f:
    envs_data = json.load(f)

# Flatten both MiniGrid and BabyAI lists into a single lookup dict
env_lookup = {}
for group in ["MiniGrid", "BabyAI"]:
    for env in envs_data.get(group, []):
        env_lookup[env["sub_link"]] = env

# Track used sub_links
used_sub_links = set()

# Replace sub_link names in curriculum with full dicts
for category in curriculum:
    new_levels = []
    for sub_link in category["levels"]:
        env = env_lookup.get(sub_link)
        if env:
            new_levels.append(env)
            used_sub_links.add(sub_link)
        else:
            new_levels.append({"sub_link": sub_link, "error": "NOT FOUND"})
    category["levels"] = new_levels

# Save the merged result
merged_output_path = Path("merged_curriculum2.json")
with open(merged_output_path, "w") as f:
    json.dump(curriculum, f, indent=2)

# Identify unused environments
all_sub_links = set(env_lookup.keys())
unused_sub_links = sorted(all_sub_links - used_sub_links)
print(unused_sub_links) # should be playground env
