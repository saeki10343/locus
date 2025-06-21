# src/generate_hunk_ids.py

import json

with open("data/hunks.json") as f:
    hunks = json.load(f)

# 各ハンクのIDを "commit_id:index" として構築
hunk_ids = [f"{h['commit_id']}:{h['index']}" for h in hunks]

with open("data/commit_ids.json", "w") as f:
    json.dump(hunk_ids, f)

print(f"Saved {len(hunk_ids)} hunk IDs to data/commit_ids.json")
