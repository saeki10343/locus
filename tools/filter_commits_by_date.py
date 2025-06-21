import json
from datetime import datetime

START_DATE = datetime(2014, 6, 25)
END_DATE = datetime(2018, 6, 30, 23, 59, 59)

def is_within_tomcat_8_0_range(date_str):
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return START_DATE <= dt <= END_DATE
    except Exception:
        return False

with open("data/commits.json") as f:
    commits = json.load(f)

filtered = [c for c in commits if is_within_tomcat_8_0_range(c.get("date", ""))]

print(f"Filtered {len(filtered)} commits (out of {len(commits)})")

with open("data/commits.json", "w") as f:
    json.dump(filtered, f, indent=2)
