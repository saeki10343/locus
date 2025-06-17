# src/evaluate_ranking.py

import json
import os
import sys
from datetime import datetime

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# allow running the script directly without PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from build_corpus import load_commit_corpus, build_tfidf_matrix

def clean(text):
    import re
    return re.sub(r"[^\w\s]", " ", text.lower())

def load_data():
    with open("data/bug_reports.json", "r") as f:
        bugs = json.load(f)

    with open("data/commit_ids.json", "r") as f:
        commit_ids = json.load(f)

    with open("data/commits.json", "r") as f:
        commits = json.load(f)

    commit_dates = [datetime.fromisoformat(c["date"]) for c in commits]
    commit_files = {c["hash"]: [d["file"] for d in c.get("diffs", [])] for c in commits}

    matrix = load_npz("data/tfidf.npz")
    _, docs = load_commit_corpus("data/commits.json")
    _, vectorizer = build_tfidf_matrix(docs)

    return bugs, commit_ids, commit_dates, commit_files, matrix, vectorizer

def evaluate(bugs, commit_ids, commit_dates, commit_files, tfidf_matrix, vectorizer, ks=(1, 5, 10)):
    """Evaluate retrieval performance for the given bug reports.

    Parameters
    ----------
    bugs : list[dict]
        Bug report objects loaded from ``bug_reports.json``.
    commit_ids : list[str]
        List of commit hashes corresponding to rows in ``tfidf_matrix``.
    commit_dates : list[datetime]
        Commit timestamps aligned with ``commit_ids``.
    commit_files : dict[str, list[str]]
        Mapping of commit hash to files modified in that commit.
    tfidf_matrix : scipy.sparse.spmatrix
        TF-IDF matrix built from commit messages and diffs.
    vectorizer : TfidfVectorizer
        Fitted vectorizer used to create the matrix.
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

    for bug in bugs:
        bug_text = clean(bug["summary"] + " " + bug.get("description", ""))
        query_vec = vectorizer.transform([bug_text])

        # temporal filter
        valid_indices = list(range(len(commit_ids)))
        if bug.get("report_time"):
            try:
                cutoff = datetime.fromisoformat(bug["report_time"].replace("Z", "+00:00"))
                valid_indices = [i for i, dt in enumerate(commit_dates) if dt <= cutoff]
            except Exception:
                pass

        if not valid_indices:
            continue

        scores = cosine_similarity(query_vec, tfidf_matrix[valid_indices]).flatten()
        ranked_idx = np.argsort(scores)[::-1]
        ranked_commits = [valid_indices[i] for i in ranked_idx]

        # map commit ranking to file ranking
        file_rank = {}
        for r, idx in enumerate(ranked_commits):
            cid = commit_ids[idx]
            for f in commit_files.get(cid, []):
                if f not in file_rank:
                    file_rank[f] = r

        # gold files from fix commits
        gold_files = []
        for fix in bug.get("fixes", []):
            cid = next((c for c in commit_ids if c.startswith(fix)), None)
            if cid:
                for f in commit_files.get(cid, []):
                    if f not in gold_files:
                        gold_files.append(f)

        ranks = [file_rank.get(f) for f in gold_files if f in file_rank]

        if ranks:
            ranks.sort()
            ap = sum([(i + 1) / (r + 1) for i, r in enumerate(ranks)]) / len(gold_files)
            rr = 1 / (ranks[0] + 1)
        else:
            ap = 0.0
            rr = 0.0

        ap_list.append(ap)
        rr_list.append(rr)

        for k in ks:
            top_files = set()
            for idx in ranked_commits[:k]:
                cid = commit_ids[idx]
                top_files.update(commit_files.get(cid, []))
            hit_dict[k].append(int(any(f in top_files for f in gold_files)))

    result = {
        "MAP": np.mean(ap_list),
        "MRR": np.mean(rr_list),
    }

    for k in ks:
        result[f"TOP@{k}"] = np.mean(hit_dict[k])

    return result

if __name__ == "__main__":
    bugs, commit_ids, commit_dates, commit_files, matrix, vectorizer = load_data()
    result = evaluate(bugs, commit_ids, commit_dates, commit_files, matrix, vectorizer, ks=(1, 5, 10))
    print("\n=== Evaluation Result ===")
    for key, val in result.items():
        print(f"{key}: {val:.4f}")
