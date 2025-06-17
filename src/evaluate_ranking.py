# src/evaluate_ranking.py

import json
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from src.build_corpus import load_commit_corpus, build_tfidf_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

def clean(text):
    import re
    return re.sub(r"[^\w\s]", " ", text.lower())

def load_data():
    with open("data/bug_reports.json", "r") as f:
        bugs = json.load(f)

    with open("data/commit_ids.json", "r") as f:
        commit_ids = json.load(f)

    matrix = load_npz("data/tfidf.npz")
    _, docs = load_commit_corpus("data/commits.json")
    _, vectorizer = build_tfidf_matrix(docs)

    return bugs, commit_ids, matrix, vectorizer

def evaluate(bugs, commit_ids, tfidf_matrix, vectorizer, ks=(1, 5, 10)):
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

    for bug in bugs:
        bug_text = clean(bug["summary"] + " " + bug.get("description", ""))
        query_vec = vectorizer.transform([bug_text])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        ranked = np.argsort(scores)[::-1]

        # 正解コミットに該当するインデックスを取得
        gold_set = set(fix[:7] for fix in bug["fixes"])
        hit_rank = []
        for rank, idx in enumerate(ranked):
            commit_prefix = commit_ids[idx][:7]
            if commit_prefix in gold_set:
                hit_rank.append(rank)

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
    bugs, commit_ids, matrix, vectorizer = load_data()
    result = evaluate(bugs, commit_ids, matrix, vectorizer, ks=(1, 5, 10))
    print("\n=== Evaluation Result ===")
    for key, val in result.items():
        print(f"{key}: {val:.4f}")
