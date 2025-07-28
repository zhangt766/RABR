# evaluate_predictions.py
import json
import numpy as np
from collections import defaultdict

# ==== Config ====
input_path = "output_predictions.json"
beta = 0.5  # F_beta weight
k = 5       # for Recall@k

# ==== F_beta Calculation ====
def f_beta_score(pred, truth, beta=1.0):
    pred_set, truth_set = set(pred), set(truth)
    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
    return precision, recall, f_beta

# ==== Recall@K ====
def recall_at_k(pred, truth, k):
    pred_topk = set(pred[:k])
    truth_set = set(truth)
    if not truth_set:
        return 0.0
    hit = len(pred_topk & truth_set)
    return hit / len(truth_set)

# ==== Load Predictions ====
with open(input_path, "r") as f:
    predictions = json.load(f)

# ==== Evaluation ====
precisions, recalls, f_scores, sizes, recall_at_ks = [], [], [], [], []

for entry in predictions:
    pred = entry["predicted_bundle"]
    truth = entry["ground_truth_bundle"]
    precision, recall, f_beta = f_beta_score(pred, truth, beta)
    r_at_k = recall_at_k(pred, truth, k)

    precisions.append(precision)
    recalls.append(recall)
    f_scores.append(f_beta)
    sizes.append(len(pred))
    recall_at_ks.append(r_at_k)

# ==== Report Results ====
print("\n📊 Evaluation Results")
print(f"Precision:       {np.mean(precisions):.4f}")
print(f"Recall:          {np.mean(recalls):.4f}")
print(f"F{beta}-score:    {np.mean(f_scores):.4f}")
print(f"Coverage@{k}:     {np.mean(recall_at_ks):.4f} (Recall@{k})")
print(f"Avg Size:        {np.mean(sizes):.2f} items per bundle")