# src/extract_commits.py

import os
from git import Repo
import json
from tqdm import tqdm

def extract_commits(repo_path, max_count=1000):
    repo = Repo(repo_path)
    commits_data = []

    for commit in tqdm(list(repo.iter_commits('main', max_count=max_count))):
        if not commit.parents:
            continue  # skip initial commit
        
        diff_data = []
        diffs = commit.diff(commit.parents[0], create_patch=True)
        for diff in diffs:
            if diff.new_file or diff.deleted_file:
                continue  # skip added/deleted files
            try:
                patch = diff.diff.decode('utf-8', errors='ignore')
                diff_data.append({
                    'file': diff.b_path,
                    'patch': patch
                })
            except Exception as e:
                continue
        
        commits_data.append({
            'hash': commit.hexsha,
            'message': commit.message.strip(),
            'author': commit.author.name,
            'date': commit.committed_datetime.isoformat(),
            'diffs': diff_data
        })

    return commits_data

if __name__ == "__main__":
    REPO_PATH = "../tomcat"  # あなたの環境に合わせて変更
    OUTPUT_FILE = "data/commits.json"

    os.makedirs("data", exist_ok=True)
    commits = extract_commits(REPO_PATH, max_count=1000)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(commits, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(commits)} commits to {OUTPUT_FILE}")