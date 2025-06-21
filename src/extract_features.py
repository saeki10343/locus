import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import numpy as np

NL_FILE = "data/bug_reports.json"
CE_FILE = "data/hunk_corpus.json"
VOCAB_FILE = "data/tfidf_vocab.json"
NL_MATRIX_FILE = "data/nl_tfidf.npz"
CE_MATRIX_FILE = "data/ce_tfidf.npz"

def load_texts(json_path, key):
    with open(json_path) as f:
        items = json.load(f)
    texts = []
    for item in items:
        value = item.get(key)
        if isinstance(value, list):
            texts.append(" ".join(value))
        elif isinstance(value, str):
            texts.append(value)
        else:
            texts.append("")
    return texts

def save_sparse_matrix(filename, matrix):
    sparse.save_npz(filename, matrix)

def convert_vocab_to_serializable(vocab):
    return {str(k): int(v) for k, v in vocab.items()}

def main():
    print("Loading data...")
    nl_texts = load_texts(NL_FILE, "summary")  # could use description too
    ce_texts = load_texts(CE_FILE, "ce")

    print("Fitting shared TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=10000)
    vectorizer.fit(nl_texts + ce_texts)

    print("Transforming NL texts...")
    nl_matrix = vectorizer.transform(nl_texts)
    print("Transforming CE texts...")
    ce_matrix = vectorizer.transform(ce_texts)

    print("Saving TF-IDF matrices and vocab")
    save_sparse_matrix(NL_MATRIX_FILE, nl_matrix)
    save_sparse_matrix(CE_MATRIX_FILE, ce_matrix)
    with open(VOCAB_FILE, "w") as f:
        json.dump(convert_vocab_to_serializable(vectorizer.vocabulary_), f)

    print("Done.")

if __name__ == "__main__":
    main()
