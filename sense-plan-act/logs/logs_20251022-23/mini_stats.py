# %% imports
import json
import pandas as pd
from pathlib import Path

folder = Path(__file__).parent

# %% Load metrics and extract level summaries
metrics = []
with open(folder / "metrics.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            metrics.append(json.loads(line))

# Filter to level_summary entries only
summaries = [m for m in metrics if m.get("type") == "level_summary"]
print(f"Total level summaries: {len(summaries)}")

# Build dataframe
rows = []
for s in summaries:
    row = {
        "level_name": s["level"]["level_name"],
        "outcome": s["outcome"],
        **s["counters"]
    }
    rows.append(row)

df = pd.DataFrame(rows)
print(df.head())

# %% Stats

success = df[df["outcome"] == "success"]
total_success = len(success)
total_levels = len(df)

print(f"\n{'='*60}")
print(f"Total levels: {total_levels}, Successful: {total_success}")
print(f"{'='*60}\n")

# 1. Out of levels with coder_first>=1, how many have coder_semantic=0 AND success?
has_coder_first = df[df["coder_first"] >= 1]
first_hit_success = has_coder_first[(has_coder_first["coder_semantic"] == 0) & (has_coder_first["outcome"] == "success")]
print(f"1. Coder first-hit solves (out of levels with coder_first>=1):")
print(f"   {len(first_hit_success)} / {len(has_coder_first)} levels with coder_first ({100*len(first_hit_success)/len(has_coder_first):.1f}%)")

# 2. P(success | coder_semantic>0)
has_semantic = df[df["coder_semantic"] > 0]
semantic_success = has_semantic[has_semantic["outcome"] == "success"]
print(f"\n2. P(success | coder_semantic>0):")
print(f"   {len(semantic_success)} / {len(has_semantic)} levels with semantic repair ({100*len(semantic_success)/len(has_semantic):.1f}%)")

# 3. P(success | planner_replan>0)
has_replan = df[df["planner_replan"] > 0]
replan_success = has_replan[has_replan["outcome"] == "success"]
print(f"\n3. P(success | planner_replan>0):")
print(f"   {len(replan_success)} / {len(has_replan)} levels with replan ({100*len(replan_success)/len(has_replan):.1f}%)")

# 4. Does reuse help?
# Compare spca_rounds between successful levels with/without reuse
success_reuse_yes = success[success["planner_reuse"] > 0]
success_reuse_no = success[success["planner_reuse"] == 0]

print(f"\n4. Does reuse help? (comparing successful levels only)")
print(f"   Success with reuse (planner_reuse>0): n={len(success_reuse_yes)}")
print(f"      mean spca_rounds: {success_reuse_yes['spca_rounds'].mean():.2f}, median: {success_reuse_yes['spca_rounds'].median():.1f}")
print(f"   Success without reuse (planner_reuse=0): n={len(success_reuse_no)}")
print(f"      mean spca_rounds: {success_reuse_no['spca_rounds'].mean():.2f}, median: {success_reuse_no['spca_rounds'].median():.1f}")

# P(spca_rounds==1 | success AND planner_reuse>0)
efficient_reuse = success_reuse_yes[success_reuse_yes["spca_rounds"] == 1]
print(f"   P(spca_rounds==1 | success & planner_reuse>0): {len(efficient_reuse)} / {len(success_reuse_yes)} ({100*len(efficient_reuse)/len(success_reuse_yes):.1f}%)")

# 5. Is heavy repair due to few nasty levels?
# Per level, sum coder_semantic, sort by total, top 10
semantic_per_level = df.groupby("level_name")["coder_semantic"].sum().sort_values(ascending=False)
total_semantic = semantic_per_level.sum()

print(f"\n5. Is heavy repair due to few nasty levels?")
print(f"   Total coder_semantic repairs across all levels: {int(total_semantic)}")
print(f"\n   Top 10 levels by coder_semantic:")

top10 = semantic_per_level.head(10)
cumsum = 0
for i, (level, count) in enumerate(top10.items(), 1):
    cumsum += count
    pct = 100 * count / total_semantic
    cumsum_pct = 100 * cumsum / total_semantic
    print(f"   {i:2}. {level}: {int(count)} ({pct:.1f}%, cumulative: {cumsum_pct:.1f}%)")

print(f"\n   Top 10 levels account for {100*top10.sum()/total_semantic:.1f}% of all semantic repairs")





# ============================================================
# Total levels: 81, Successful: 76
# ============================================================

# 1. Coder first-hit solves (out of levels with coder_first>=1):
#    6 / 12 levels with coder_first (50.0%)

# 2. P(success | coder_semantic>0):
#    21 / 26 levels with semantic repair (80.8%)

# 3. P(success | planner_replan>0):
#    9 / 14 levels with replan (64.3%)

# 4. Does reuse help? (comparing successful levels only)
#    Success with reuse (planner_reuse>0): n=66
#       mean spca_rounds: 1.20, median: 1.0
#    Success without reuse (planner_reuse=0): n=10
#       mean spca_rounds: 1.10, median: 1.0
#    P(spca_rounds==1 | success & planner_reuse>0): 58 / 66 (87.9%)

# 5. Is heavy repair due to few nasty levels?
#    Total coder_semantic repairs across all levels: 213

#    Top 10 levels by coder_semantic:
#     1. BabyAI-Unlock-v0: 25 (11.7%, cumulative: 11.7%)
#     2. BabyAI-UnlockPickupDist-v0: 25 (11.7%, cumulative: 23.5%)
#     3. MiniGrid-GoToDoor-8x8-v0: 25 (11.7%, cumulative: 35.2%)
#     4. MiniGrid-UnlockPickup-v0: 25 (11.7%, cumulative: 46.9%)
#     5. BabyAI-Open-v0: 18 (8.5%, cumulative: 55.4%)
#     6. MiniGrid-GoToDoor-5x5-v0: 15 (7.0%, cumulative: 62.4%)
#     7. BabyAI-PickupAbove-v0: 14 (6.6%, cumulative: 69.0%)
#     8. BabyAI-UnlockToUnlock-v0: 8 (3.8%, cumulative: 72.8%)
#     9. BabyAI-KeyInBox-v0: 8 (3.8%, cumulative: 76.5%)
#    10. MiniGrid-MultiRoom-N4-S5-v0: 7 (3.3%, cumulative: 79.8%)

#    Top 10 levels account for 79.8% of all semantic repairs