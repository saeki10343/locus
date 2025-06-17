import json
import os
import re
from typing import Dict, List

import git
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def fetch_bug_report(bug_id: str) -> Dict:
    """Fetch summary and description for a Bugzilla bug."""
    url = f"https://bz.apache.org/bugzilla/show_bug.cgi?id={bug_id}"
    res = requests.get(url, timeout=30)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    summary = soup.find("span", id="short_desc_nonedit_display")
    description = soup.find("pre", class_="bz_comment_text")
    return {
        "id": f"BUG-{bug_id}",
        "summary": summary.text.strip() if summary else "",
        "description": description.text.strip() if description else "",
    }


def scan_repository(repo_path: str) -> Dict[str, List[str]]:
    """Scan git history and return mapping bug_id -> list of commits."""
    repo = git.Repo(repo_path)
    bug_map: Dict[str, List[str]] = {}
    pattern = re.compile(r"bug\s*(\d{3,6})", re.IGNORECASE)
    for commit in tqdm(repo.iter_commits("main")):
        matches = pattern.findall(commit.message)
        for bid in matches:
            bug_map.setdefault(bid, []).append(commit.hexsha)
    return bug_map


def main():
    repo_path = os.environ.get("TOMCAT_REPO", "tomcat")
    bug_map = scan_repository(repo_path)

    reports = []
    for bug_id, commits in tqdm(bug_map.items()):
        try:
            br = fetch_bug_report(bug_id)
            br["fixes"] = [c[:7] for c in commits]
            reports.append(br)
        except Exception as e:
            print(f"Failed to fetch BUG-{bug_id}: {e}")

    os.makedirs("data", exist_ok=True)
    with open("data/bug_reports.json", "w") as f:
        json.dump(reports, f, indent=2)

    print(f"Saved {len(reports)} bug reports to data/bug_reports.json")


if __name__ == "__main__":
    main()
