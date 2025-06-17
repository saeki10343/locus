# src/compute_similarity.py

import json
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import load_npz
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from build_corpus import load_commit_corpus, build_tfidf_matrix

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

def load_bug_reports(path):
    with open(path, 'r', encoding='utf-8') as f:
        bugs = json.load(f)
    return [(bug['id'], clean_text(bug['summary'] + ' ' + bug.get('description', ''))) for bug in bugs]

def rank_commits_for_bug(bug_text, vectorizer, tfidf_matrix, commit_ids, top_k=10):
    query_vector = vectorizer.transform([bug_text])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    return [(commit_ids[i], similarities[i]) for i in ranked_indices]

if __name__ == "__main__":
    BUG_REPORT_FILE = "data/bug_reports.json"  # バグ報告を事前に用意する必要あり
    TFIDF_MATRIX_FILE = "data/tfidf.npz"
    COMMIT_IDS_FILE = "data/commit_ids.json"

    bug_reports = load_bug_reports(BUG_REPORT_FILE)
    tfidf_matrix = load_npz(TFIDF_MATRIX_FILE)

    with open(COMMIT_IDS_FILE, 'r') as f:
        commit_ids = json.load(f)

    # vectorizer再構築
    from src.build_corpus import load_commit_corpus, build_tfidf_matrix
    _, docs = load_commit_corpus("data/commits.json")
    _, vectorizer = build_tfidf_matrix(docs)

    for bug_id, bug_text in bug_reports:
        top_results = rank_commits_for_bug(bug_text, vectorizer, tfidf_matrix, commit_ids)
        print(f"\nBug ID: {bug_id}")
        for cid, score in top_results:
            print(f"  {cid[:8]}...  Score: {score:.4f}")
