import json
import re
import os
from tqdm import tqdm

HUNK_HEADER_RE = re.compile(r"^@@ -\d+(,\d+)? \+\d+(,\d+)? @@")


def split_patch_into_hunks(patch):
    lines = patch.splitlines()
    hunks = []
    current_hunk = []
    for line in lines:
        if HUNK_HEADER_RE.match(line):
            if current_hunk:
                hunks.append(current_hunk)
            current_hunk = [line]
        elif current_hunk:
            current_hunk.append(line)
    if current_hunk:
        hunks.append(current_hunk)
    return hunks


def is_valid_hunk(hunk_lines):
    for line in hunk_lines:
        content = line.lstrip()[1:].strip()
        if not content:
            continue
        if content.startswith("*") and content.endswith("="):
            continue
        if content.startswith("//") or content.startswith("/*") or content.startswith("*"):
            continue
        if content.startswith("#"):
            continue
        return True
    return False


def extract_hunks_from_commits(commits_file, output_file):
    with open(commits_file) as f:
        commits = json.load(f)

    hunk_id = 0
    hunk_data = []

    for commit in tqdm(commits, desc="Extracting hunks"):
        commit_id = commit.get("hash") or commit.get("commit_id")
        for diff in commit.get("diffs", []):
            patch = diff.get("patch")
            file_path = diff.get("file") or diff.get("new_path") or diff.get("old_path")
            if not patch or not file_path:
                continue
            if file_path.endswith(".xml") or file_path.startswith("webapps/"):
                continue
            hunks = split_patch_into_hunks(patch)
            for i, hunk in enumerate(hunks):
                if not is_valid_hunk(hunk):
                    continue
                hunk_text = "\n".join(hunk)
                hunk_data.append({
                    "hunk_id": f"{commit_id}_{hunk_id}",
                    "commit_id": commit_id,
                    "file_path": file_path,
                    "hunk": hunk_text,
                    "index": i
                })
                hunk_id += 1

    with open(output_file, "w") as f:
        json.dump(hunk_data, f, indent=2)

    print(f"Saved {len(hunk_data)} hunks to {output_file}")


if __name__ == "__main__":
    extract_hunks_from_commits("data/commits.json", "data/hunks.json")
