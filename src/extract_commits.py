import os
import json
import re
from argparse import ArgumentParser
from git import Repo
from tqdm import tqdm


def extract_commits(repo_path: str, branch: str = "main", max_count=None, since=None, until=None, skip_tests=True):
    repo = Repo(repo_path)
    commits_data = []
    iterator = repo.iter_commits(branch, max_count=max_count, since=since, until=until)
    for commit in tqdm(list(iterator)):
        if not commit.parents:
            continue
        diffs = commit.diff(commit.parents[0], create_patch=True)
        diff_data = []
        for diff in diffs:
            if diff.new_file or diff.deleted_file:
                continue
            if skip_tests and diff.b_path and "/test/" in diff.b_path.lower():
                continue
            try:
                patch = diff.diff.decode("utf-8", errors="ignore")
                diff_data.append({"file": diff.b_path, "patch": patch})
            except Exception:
                continue
        commits_data.append({
            "hash": commit.hexsha,
            "message": commit.message.strip(),
            "author": commit.author.name,
            "date": commit.committed_datetime.isoformat(),
            "diffs": diff_data,
        })
    return commits_data


def main():
    ap = ArgumentParser(description="Extract commit data from a git repository")
    ap.add_argument("repo", help="Path to git repository")
    ap.add_argument("output", help="Path to output JSON file")
    ap.add_argument("--branch", default="main", help="Branch to scan")
    ap.add_argument("--max-count", type=int, default=None, help="Limit number of commits")
    ap.add_argument("--since", help="Only include commits after this date (YYYY-MM-DD)")
    ap.add_argument("--until", help="Only include commits before this date (YYYY-MM-DD)")
    ap.add_argument("--no-skip-tests", action="store_true", help="Do not filter test files")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    commits = extract_commits(
        args.repo,
        branch=args.branch,
        max_count=args.max_count,
        since=args.since,
        until=args.until,
        skip_tests=not args.no_skip_tests,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(commits, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(commits)} commits to {args.output}")


if __name__ == "__main__":
    main()
