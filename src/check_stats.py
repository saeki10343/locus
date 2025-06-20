import json

def count_bugs(bug_report_path):
    with open(bug_report_path, encoding="utf-8") as f:
        bugs = json.load(f)
    return len(bugs)

def count_unique_files(commit_features_path):
    with open(commit_features_path, encoding="utf-8") as f:
        feats = json.load(f)
    file_set = set()
    for item in feats:
        for file in item.get("files", []):
            file_set.add(file)
    return len(file_set)

def count_commits(commits_path):
    with open(commits_path, encoding="utf-8") as f:
        commits = json.load(f)
    return len(commits)

if __name__ == "__main__":
    bug_report_path = "data/bug_reports.json"
    commit_features_path = "data/commit_features.json"
    commits_path = "data/commits.json"

    num_bugs = count_bugs(bug_report_path)
    num_files = count_unique_files(commit_features_path)
    num_commits = count_commits(commits_path)

    print("=== Locus Reproduction Stats ===")
    print(f"# Bugs:    {num_bugs}")
    print(f"# Files:   {num_files}")
    print(f"# Commits: {num_commits}")
