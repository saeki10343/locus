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

def evaluate(bugs, commit_ids, tfidf_matrix, vectorizer, k=10):
    ap_list = []
    rr_list = []
    hitk_list = []

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

        # 評価指標
        if hit_rank:
            ap = sum([(i + 1) / (r + 1) for i, r in enumerate(hit_rank)]) / len(hit_rank)
            rr = 1 / (hit_rank[0] + 1)
            hitk = int(any(r < k for r in hit_rank))
        else:
            ap = 0.0
            rr = 0.0
            hitk = 0

        ap_list.append(ap)
        rr_list.append(rr)
        hitk_list.append(hitk)

    return {
        "MAP": np.mean(ap_list),
        "MRR": np.mean(rr_list),
        f"Hit@{k}": np.mean(hitk_list)
    }

if __name__ == "__main__":
    bugs, commit_ids, matrix, vectorizer = load_data()
    result = evaluate(bugs, commit_ids, matrix, vectorizer, k=10)
    print("\n=== Evaluation Result ===")
    for key, val in result.items():
        print(f"{key}: {val:.4f}")
