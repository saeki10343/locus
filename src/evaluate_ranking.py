# src/evaluate_hunks.py

import json
import numpy as np
from collections import defaultdict

def load_data():
    with open("data/similarity_scores.json") as f:
        scores = json.load(f)
    with open("data/fix_hunk_map.json") as f:
        fix_hunk_ids = json.load(f)
    return scores, fix_hunk_ids

def evaluate(scores, fix_hunk_ids, ks=(1, 5, 10)):
    ap_list = []
    rr_list = []
    hit_dict = {k: [] for k in ks}

    for bug_id, predictions in scores.items():
        gold_hunks = set(fix_hunk_ids.get(bug_id, []))
        if not gold_hunks:
            continue

        predicted_ids = [item.get("commit_id") for item in predictions if "commit_id" in item]
        hit_ranks = [i for i, hid in enumerate(predicted_ids) if hid in gold_hunks]

        if hit_ranks:
            ap = sum([(i + 1) / (r + 1) for i, r in enumerate(hit_ranks)]) / len(hit_ranks)
            rr = 1 / (hit_ranks[0] + 1)
        else:
            ap = 0.0
            rr = 0.0

        ap_list.append(ap)
        rr_list.append(rr)
        for k in ks:
            hit_dict[k].append(int(any(r < k for r in hit_ranks)))

    result = {
        "MAP": float(np.mean(ap_list)) if ap_list else 0.0,
        "MRR": float(np.mean(rr_list)) if rr_list else 0.0,
    }
    for k in ks:
        result[f"TOP@{k}"] = float(np.mean(hit_dict[k])) if hit_dict[k] else 0.0
    return result

if __name__ == "__main__":
    scores, fix_hunk_ids = load_data()
    result = evaluate(scores, fix_hunk_ids)

    print("\n=== Evaluation Result (Hunk Level) ===")
    for key, val in result.items():
        print(f"{key}: {val:.4f}")
