# src/evaluate_ranking.py

import json
import os
import re
import sys

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# allow running the script directly without PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from build_corpus import load_commit_corpus, build_tfidf_matrix

CODE_TOKEN_RE = re.compile(r'[A-Za-z_]*[A-Z_][A-Za-z0-9_]*')

def emphasize_code_tokens(text: str, weight: int = 5) -> str:
    tokens = text.split()
    out = []
    for t in tokens:
        if CODE_TOKEN_RE.search(t):
            out.extend([t] * weight)
        else:
            out.append(t)
    return ' '.join(out)

def load_commit_boost(path: str, beta: float = 0.1):
    with open(path, 'r') as f:
        feats = json.load(f)

    file_freq = {}
    for item in feats:
        for fp in item.get('files', []):
            file_freq[fp] = file_freq.get(fp, 0) + 1

    commit_boost = {}
    for item in feats:
        freq = sum(file_freq.get(fp, 0) for fp in item.get('files', []))
        commit_boost[item['commit_id']] = freq

    max_freq = max(commit_boost.values()) if commit_boost else 1
    for cid in commit_boost:
        commit_boost[cid] = 1.0 + beta * (commit_boost[cid] / max_freq)
    return commit_boost

def clean(text):
    import re
    return re.sub(r"[^\w\s]", " ", text.lower())

def load_data():
    with open("data/bug_reports.json", "r") as f:
        bugs = json.load(f)

    with open("data/commit_ids.json", "r") as f:
        commit_ids = json.load(f)

    matrix = load_npz("data/tfidf.npz")
    _, docs, dates = load_commit_corpus("data/commits.json")
    _, vectorizer = build_tfidf_matrix(docs)

    commit_boost = load_commit_boost("data/commit_features.json")

    with open("data/commit_features.json", "r") as f:
        feats = json.load(f)
        commit_files = {}
        for item in feats:
            cid = item["commit_id"]
            files = item.get("files", [])
            commit_files[cid] = files
            commit_files[cid[:7]] = files

    return bugs, commit_ids, dates, matrix, vectorizer, commit_boost, commit_files

from datetime import datetime

def parse_time(ts: str):
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        for fmt in ("%Y-%m-%d %H:%M:%S %Z", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(ts, fmt)
            except Exception:
                continue
    return None


def evaluate(bugs, commit_ids, commit_dates, tfidf_matrix, vectorizer, commit_boost, commit_files, ks=(1, 5, 10)):
    """Evaluate retrieval performance for the given bug reports.

    Parameters
    ----------
    bugs : list[dict]
        Bug report objects loaded from ``bug_reports.json``.
    commit_ids : list[str]
        List of commit hashes corresponding to rows in ``tfidf_matrix``.
    commit_dates : list[str]
        Commit timestamp strings aligned with ``commit_ids``.
    tfidf_matrix : scipy.sparse.spmatrix
        TF-IDF matrix built from commit messages and diffs.
    vectorizer : TfidfVectorizer
        Fitted vectorizer used to create the matrix.
    commit_files : dict[str, list[str]]
        Mapping of commit id (or 7-char prefix) to changed file paths.
    ks : tuple[int]
        Cutoff values for TOP@K style metrics.

    Returns
    -------
    dict
        Mapping of metric name to score.
    """

    ap_list = []
    rr_list = []
    hit_dict = {k: [] for k in ks}

    commit_times = [parse_time(ts) if ts else None for ts in commit_dates]

    for bug in bugs:
        bug_text = clean(bug["summary"] + " " + bug.get("description", ""))
        bug_text = emphasize_code_tokens(bug_text)

        bug_time = None
        for key in ("date", "created", "creation_time"):
            if key in bug:
                bug_time = parse_time(bug[key])
                break

        # filter commits by time if timestamp is available
        valid_idx = list(range(len(commit_ids)))
        if bug_time:
            valid_idx = [i for i, t in enumerate(commit_times) if t and t <= bug_time]

        if not valid_idx:
            continue

        sub_matrix = tfidf_matrix[valid_idx]
        sub_ids = [commit_ids[i] for i in valid_idx]
        sub_boost = np.array([commit_boost.get(cid, 1.0) for cid in sub_ids])

        query_vec = vectorizer.transform([bug_text])
        scores = cosine_similarity(query_vec, sub_matrix).flatten()
        scores = scores * sub_boost
        ranked = np.argsort(scores)[::-1]

        gold_files = set()
        for fix in bug.get("fixes", []):
            for fp in commit_files.get(fix, []):
                gold_files.add(fp)

        ranked_files = []
        for idx in ranked:
            cid = sub_ids[idx]
            ranked_files.extend(commit_files.get(cid, []))

        hit_rank = [i for i, fp in enumerate(ranked_files) if fp in gold_files]

        # metrics per bug
        if hit_rank:
            ap = sum([(i + 1) / (r + 1) for i, r in enumerate(hit_rank)]) / len(hit_rank)
            rr = 1 / (hit_rank[0] + 1)
        else:
            ap = 0.0
            rr = 0.0

        ap_list.append(ap)
        rr_list.append(rr)

        for k in ks:
            hit_dict[k].append(int(any(r < k for r in hit_rank)))

    result = {
        "MAP": np.mean(ap_list),
        "MRR": np.mean(rr_list),
    }

    for k in ks:
        result[f"TOP@{k}"] = np.mean(hit_dict[k])

    return result

if __name__ == "__main__":
    bugs, commit_ids, dates, matrix, vectorizer, commit_boost, commit_files = load_data()
    result = evaluate(bugs, commit_ids, dates, matrix, vectorizer, commit_boost, commit_files, ks=(1, 5, 10))
    print("\n=== Evaluation Result ===")
    for key, val in result.items():
        print(f"{key}: {val:.4f}")
