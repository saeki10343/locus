import json
import re
import requests
from tqdm import tqdm

def fetch_bug_report_rest(bug_id):
    url = f"https://bz.apache.org/bugzilla/rest.cgi/bug/{bug_id}?Bugzilla_api_key={'azVGl7F6Kj4iw3xIn4DiMbtqYrfPfznyOTsF0BJW'}"
    res = requests.get(url, timeout=10)
    try:
        data = res.json()
    except json.JSONDecodeError:
        raise Exception(f"Bug {bug_id} returned invalid JSON (status={res.status_code})")

    if "bugs" not in data or not data["bugs"]:
        raise Exception(f"Bug {bug_id} not found")

    bug = data["bugs"][0]
    if bug["status"] != "RESOLVED" or bug.get("resolution") != "FIXED":
        raise Exception(f"Bug {bug_id} is not RESOLVED FIXED")
    if not bug["version"].startswith("8.0"):
        raise Exception(f"Bug {bug_id} is not Tomcat verison 8.0")

    return {
        "id": f"BUG-{bug_id}",
        "product": bug["product"],
        "version": bug["version"],
        "summary": bug["summary"],
        "description": "",  # Bugzilla REST API には description が含まれない
        "created": bug.get("creation_time"),
        "fixes": [],  # 後で付与
    }

def collect_from_commit_log(commits_file):
    with open(commits_file) as f:
        commits = json.load(f)

    bug_map = {}
    for c in commits:
        commit_id = c.get("hash") or c.get("commit_id")
        diffs = c.get("diffs", [])
        for diff in diffs:
            patch = diff.get("patch", "").lower()
            matches = re.findall(r"<bug>(\d+)</bug>", patch)
            for bid in matches:
                if bid not in bug_map:
                    bug_map[bid] = []
                bug_map[bid].append(commit_id)

    return bug_map

if __name__ == "__main__":
    bug_map = collect_from_commit_log("data/commits.json")

    result = []
    for bug_id, commit_ids in tqdm(bug_map.items(), desc="Fetching bugs"):
        try:
            bug = fetch_bug_report_rest(bug_id)
            bug["fixes"] = [cid[:7] for cid in commit_ids]
            result.append(bug)
        except Exception as e:
            print(f"Skipped BUG-{bug_id}: {e}")

    with open("data/bug_reports.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved {len(result)} RESOLVED FIXED bugs to data/bug_reports.json")