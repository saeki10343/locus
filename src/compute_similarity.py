# src/compute_similarity.py

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import load_npz

def load_vectorizer(vocab_file):
    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", vocabulary=vocab)
    return vectorizer

def load_commit_boost(path, beta=0.1):
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

def rank_commits(bug_vector, ce_matrix, commit_ids, commit_boost, top_k=10):
    similarities = cosine_similarity(bug_vector, ce_matrix).flatten()
    boost = np.array([
        commit_boost.get(cid.split(":")[0], 1.0)  # ハンクIDのコミット部分だけ使う
        for cid in commit_ids
    ])
    scores = similarities * boost
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(commit_ids[i], float(scores[i])) for i in top_indices]

def main():
    bug_report_file = "data/bug_reports.json"
    commit_ids_file = "data/commit_ids.json"
    ce_matrix_file = "data/ce_tfidf.npz"
    nl_matrix_file = "data/nl_tfidf.npz"
    vocab_file = "data/tfidf_vocab.json"
    commit_boost_file = "data/commit_features.json"

    print("Loading data...")
    with open(bug_report_file, "r") as f:
        bug_reports = json.load(f)

    with open(commit_ids_file, "r") as f:
        commit_ids = json.load(f)

    ce_matrix = load_npz(ce_matrix_file)
    nl_matrix = load_npz(nl_matrix_file)
    commit_boost = load_commit_boost(commit_boost_file)
    vectorizer = load_vectorizer(vocab_file)

    print("Computing similarity...")
    results = {}
    for i, bug in enumerate(bug_reports):
        bug_id = bug["id"]
        bug_vector = nl_matrix[i]
        ranked = rank_commits(bug_vector, ce_matrix, commit_ids, commit_boost, top_k=10)
        results[bug_id] = [{"commit_id": cid, "score": score} for cid, score in ranked]

    with open("data/similarity_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved ranked similarity results to data/similarity_scores.json")

if __name__ == "__main__":
    main()
