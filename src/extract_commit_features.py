# src/extract_commit_features.py

import json
import os
from argparse import ArgumentParser

def extract_features(commits):
    features = []
    for commit in commits:
        commit_id = commit["hash"]
        files = []
        keywords = set()
        for diff in commit.get("diffs", []):
            if "file" in diff:
                files.append(diff["file"])
            patch = diff.get("patch", "")
            for kw in ["if", "for", "while", "null", "try", "catch"]:
                if kw in patch:
                    keywords.add(kw)
        features.append({
            "commit_id": commit_id,
            "files": files,
            "keywords": list(keywords)
        })
    return features

def main():
    ap = ArgumentParser()
    ap.add_argument("input", help="Path to commits.json")
    ap.add_argument("output", help="Path to output commit_features.json")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        commits = json.load(f)

    features = extract_features(commits)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    print(f"Extracted features for {len(features)} commits")

if __name__ == "__main__":
    main()
