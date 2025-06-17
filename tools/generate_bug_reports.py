import json
import re
import requests
from bs4 import BeautifulSoup

def fetch_bug_report(bug_id):
    url = f"https://bz.apache.org/bugzilla/show_bug.cgi?id={bug_id}"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')

    summary = soup.find("span", id="short_desc_nonedit_display")
    description = soup.find("pre", class_="bz_comment_text")

    return {
        "id": f"BUG-{bug_id}",
        "summary": summary.text.strip() if summary else "",
        "description": description.text.strip() if description else ""
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
    for bug_id, commit_ids in list(bug_map.items())[:20]:
        try:
            print(f"Fetching BUG-{bug_id}")
            bug = fetch_bug_report(bug_id)
            bug["fixes"] = [cid[:7] for cid in commit_ids]
            result.append(bug)
        except Exception as e:
            print(f"Failed to fetch BUG-{bug_id}: {e}")

    with open("data/bug_reports.json", "w") as f:
        json.dump(result, f, indent=2)
