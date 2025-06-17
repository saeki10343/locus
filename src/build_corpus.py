# src/build_corpus.py

import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from diff_features import extract_features_from_patch

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

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

def load_commit_corpus(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        commits = json.load(f)

    documents = []
    ids = []
    dates = []

    for commit in commits:
        # message + hunk patch 全体を1つの document にする
        full_text = emphasize_code_tokens(commit['message'])
        for diff in commit['diffs']:
            patch = diff['patch']
            full_text += ' ' + patch
            # add extracted features from the patch with extra weight
            full_text += ' ' + extract_features_from_patch(patch, weight=5)
        full_text = clean_text(full_text)

        documents.append(full_text)
        ids.append(commit['hash'])
        dates.append(commit.get('date'))

    return ids, documents, dates

def build_tfidf_matrix(documents):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer

if __name__ == "__main__":
    input_file = "data/commits.json"
    output_matrix = "data/tfidf.npz"
    output_ids = "data/commit_ids.json"

    ids, docs, dates = load_commit_corpus(input_file)
    tfidf_matrix, vectorizer = build_tfidf_matrix(docs)

    # 保存
    from scipy.sparse import save_npz
    save_npz(output_matrix, tfidf_matrix)

    with open(output_ids, 'w') as f:
        json.dump(ids, f)

    with open("data/commit_dates.json", 'w') as f:
        json.dump(dates, f)

    print(f"TF-IDF matrix saved to {output_matrix}")
