# tools/filter_commits_by_date.py

import json
from datetime import datetime, timezone
from dateutil import parser

START_DATE = datetime(2014, 6, 25, tzinfo=timezone.utc)
END_DATE = datetime(2018, 6, 30, 23, 59, 59, tzinfo=timezone.utc)

def in_target_range(date_str):
    try:
        dt = parser.isoparse(date_str)
        return START_DATE <= dt <= END_DATE
    except Exception:
        return False

with open("data/commits-8.5.x.json") as f:
    commits = json.load(f)

filtered = [c for c in commits if in_target_range(c.get("date", ""))]

print(f"Filtered {len(filtered)} commits (out of {len(commits)})")

with open("data/commits.json", "w") as f:
    json.dump(filtered, f, indent=2)
