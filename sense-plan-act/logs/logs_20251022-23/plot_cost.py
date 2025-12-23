# %%
import pandas as pd
import json
from pathlib import Path

folder = Path(__file__).parent

# %%
# Load metrics.jsonl and extract timestamps per day
metrics = []
with open(folder / "metrics.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            metrics.append(json.loads(line))

# Extract timestamps and group by date
# Strip trailing Z since timestamps have both +00:00 and Z
timestamps = [m["timestamp"].rstrip("Z") for m in metrics if "timestamp" in m]
timestamps = pd.to_datetime(timestamps, utc=True)

# Group by date and get min/max per day
ts_df = pd.DataFrame({"ts": timestamps})
ts_df["date"] = ts_df["ts"].dt.date
day_ranges = ts_df.groupby("date")["ts"].agg(["min", "max"])
print("Time ranges from metrics.jsonl:")
print(day_ranges)
print()

# %%
# Load both CSV files
csv1 = pd.read_csv(folder / "completions_usage_2025-10-22_2025-10-22.csv")
csv2 = pd.read_csv(folder / "completions_usage_2025-10-23_2025-10-23.csv")
df = pd.concat([csv1, csv2], ignore_index=True)

# Parse timestamps
df["start_ts"] = pd.to_datetime(df["start_time_iso"], utc=True)

# %%
# Filter to only include rows within the metrics time ranges
mask = pd.Series(False, index=df.index)
for date, row in day_ranges.iterrows():
    day_mask = (df["start_ts"] >= row["min"]) & (df["start_ts"] <= row["max"])
    mask = mask | day_mask

df_filtered = df[mask].copy()
print(f"Rows before filter: {len(df)}, after filter: {len(df_filtered)}")
print()

# %%
# Sum tokens per model
summary = df_filtered.groupby("model")[["input_tokens", "output_tokens"]].sum()
print("Token usage per model:")
print(summary)
print()

# Token usage per model:
#                     input_tokens  output_tokens
# model                                          
# codex-mini-latest      1834115.0      1074050.0
# o3-2025-04-16           557646.0       188480.0
# o4-mini-2025-04-16      392107.0       113731.0

# %%
# Cost calculation ($ per million tokens)
costs = {
    "o4-mini-2025-04-16": (1.1, 4.4),
    "o3-2025-04-16": (2, 8),
    "codex-mini-latest": (1.5, 6),
}

summary["input_rate"] = summary.index.map(lambda m: costs.get(m, (0, 0))[0])
summary["output_rate"] = summary.index.map(lambda m: costs.get(m, (0, 0))[1])
summary["input_cost"] = summary["input_tokens"] / 1_000_000 * summary["input_rate"]
summary["output_cost"] = summary["output_tokens"] / 1_000_000 * summary["output_rate"]
summary["total_cost"] = summary["input_cost"] + summary["output_cost"]

print("Cost per model:")
print(summary[["input_cost", "output_cost", "total_cost"]])
print(f"\nTotal cost: ${summary['total_cost'].sum():.4f}")

# Cost per model:
#                     input_cost  output_cost  total_cost
# model                                                  
# codex-mini-latest     2.751173     6.444300    9.195473
# o3-2025-04-16         1.115292     1.507840    2.623132
# o4-mini-2025-04-16    0.431318     0.500416    0.931734

# Total cost: $12.7503