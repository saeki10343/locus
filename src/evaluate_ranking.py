# src/evaluate_ranking.py

import json
import os
import sys

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# allow running the script directly without PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from build_corpus import load_commit_corpus, build_tfidf_matrix
from datetime import datetime

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
    commit_meta = {c["hash"]: c for c in commits}
    commit_files = {c["hash"]: [d["file"] for d in c.get("diffs", [])] for c in commits}

    matrix = load_npz("data/tfidf.npz")
    _, docs = load_commit_corpus("data/commits.json")
    _, vectorizer = build_tfidf_matrix(docs)

    return bugs, commit_ids, commit_meta, commit_files, matrix, vectorizer

def evaluate(bugs, commit_ids, commit_meta, commit_files, tfidf_matrix, vectorizer, ks=(1, 5, 10)):
    """Evaluate retrieval performance for the given bug reports.

    Parameters
    ----------
    bugs : list[dict]
        Bug report objects loaded from ``bug_reports.json``.
    commit_ids : list[str]
        List of commit hashes corresponding to rows in ``tfidf_matrix``.
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

    ap_file_list = []
    rr_file_list = []
    hit_file_dict = {k: [] for k in ks}

    for bug in bugs:
        bug_date = None
        if bug.get("date"):
            try:
                bug_date = datetime.fromisoformat(bug["date"])
            except Exception:
                bug_date = None
        bug_text = clean(bug["summary"] + " " + bug.get("description", ""))
        query_vec = vectorizer.transform([bug_text])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        ranked = np.argsort(scores)[::-1]

        # 正解コミットに該当するインデックスを取得
        gold_set = set(bug["fixes"])
        hit_rank = []
        for rank, idx in enumerate(ranked):
            cid = commit_ids[idx]
            if bug_date:
                cdate = datetime.fromisoformat(commit_meta[cid]["date"])
                if cdate >= bug_date:
                    continue
            if cid in gold_set:
                hit_rank.append(rank)

        # file-level ranking
        bug_files = set()
        for fix in bug["fixes"]:
            bug_files.update(commit_files.get(fix, []))

        file_rank = []
        seen_files = set()
        for idx in ranked:
            cid = commit_ids[idx]
            if bug_date:
                cdate = datetime.fromisoformat(commit_meta[cid]["date"])
                if cdate >= bug_date:
                    continue
            for fpath in commit_files.get(cid, []):
                if fpath not in seen_files:
                    seen_files.add(fpath)
                    file_rank.append(fpath)

        file_hit = [i for i, f in enumerate(file_rank) if f in bug_files]

        # metrics per bug
        if hit_rank:
            ap = sum([(i + 1) / (r + 1) for i, r in enumerate(hit_rank)]) / len(hit_rank)
            rr = 1 / (hit_rank[0] + 1)
        else:
            ap = 0.0
            rr = 0.0

        if file_hit:
            ap_f = sum([(i + 1) / (r + 1) for i, r in enumerate(file_hit)]) / len(file_hit)
            rr_f = 1 / (file_hit[0] + 1)
        else:
            ap_f = 0.0
            rr_f = 0.0

        ap_list.append(ap)
        rr_list.append(rr)

        ap_file_list.append(ap_f)
        rr_file_list.append(rr_f)

        for k in ks:
            hit_dict[k].append(int(any(r < k for r in hit_rank)))
            hit_file_dict[k].append(int(any(r < k for r in file_hit)))

    result = {
        "MAP": np.mean(ap_list),
        "MRR": np.mean(rr_list),
        "MAP_file": np.mean(ap_file_list),
        "MRR_file": np.mean(rr_file_list),
    }

    for k in ks:
        result[f"TOP@{k}"] = np.mean(hit_dict[k])
        result[f"TOP@{k}_file"] = np.mean(hit_file_dict[k])

    return result

if __name__ == "__main__":
    bugs, commit_ids, commit_meta, commit_files, matrix, vectorizer = load_data()
    result = evaluate(bugs, commit_ids, commit_meta, commit_files, matrix, vectorizer, ks=(1, 5, 10))
    print("\n=== Evaluation Result ===")
    for key, val in result.items():
        print(f"{key}: {val:.4f}")
