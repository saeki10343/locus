import json
from collections import defaultdict

BUG_REPORTS_FILE = "data/bug_reports.json"
HUNKS_FILE = "data/hunks.json"
OUTPUT_FILE = "data/fix_hunk_map.json"

with open(BUG_REPORTS_FILE) as f:
    bugs = json.load(f)

with open(HUNKS_FILE) as f:
    hunks = json.load(f)

# Map prefix to hunk_ids
prefix_to_hunks = defaultdict(list)
for hunk in hunks:
    hunk_id = f"{hunk['commit_id']}:{hunk['index']}"
    prefix_to_hunks[hunk['commit_id'][:7]].append(hunk_id)

fix_hunk_map = {}

for bug in bugs:
    fix_prefixes = bug.get("fixes", [])
    hunk_ids = []
    for prefix in fix_prefixes:
        hunk_ids.extend(prefix_to_hunks.get(prefix, []))
    if hunk_ids:
        fix_hunk_map[bug["id"]] = hunk_ids

with open(OUTPUT_FILE, "w") as f:
    json.dump(fix_hunk_map, f, indent=2)

print(f"Saved fix_hunk_map to {OUTPUT_FILE}")
