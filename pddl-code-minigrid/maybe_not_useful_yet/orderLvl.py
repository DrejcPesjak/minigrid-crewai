import csv, json, re
from pathlib import Path

# 1) load the CSV
csv_path = Path('levels2.csv')
with csv_path.open() as f:
    reader = csv.reader(f)
    levels = [
        {'env_id': row[0].strip().strip('"'),
         'description': row[1].strip().strip('"')}
        for row in reader
        if row and row[0].startswith('"MiniGrid') or row[0].startswith('"BabyAI')
    ]

# 2) helper to extract mission_type
def mission_type(env_id):
    # strip off MiniGrid-/BabyAI- prefix and -v*
    core = re.sub(r'^(MiniGrid|BabyAI)-', '', env_id)
    core = re.sub(r'-v\d+$', '', core)
    # remove trailing size/N suffix (e.g. S9N2, 5x5, Random, etc)
    core = re.sub(r'(S\d+(N\d+)?)$', '', core)
    # normalize to snake_case
    return re.sub(r'[-]+', '_', core).lower().rstrip('_')

# 3) difficulty key from all ints in the name (so 5x5 < 6x6 < S9N1 < S11N5 < …)
def difficulty_key(env_id):
    nums = list(map(int, re.findall(r'\d+', env_id)))
    return nums + [0]  # pad so pure‐text names don’t crash

# 4) group, sort, duplicate
dup_count = 4
curriculum = {}
for lvl in levels:
    m = mission_type(lvl['env_id'])
    curriculum.setdefault(m, []).append(lvl)

for m, lvls in curriculum.items():
    lvls.sort(key=lambda L: difficulty_key(L['env_id']))
    expanded = []
    for L in lvls:
        expanded.append(L)
        if 'random' in L['env_id'].lower():
            expanded += [L.copy() for _ in range(dup_count)]
    curriculum[m] = expanded

# 5) dump to JSON
print(json.dumps(curriculum, indent=2))
