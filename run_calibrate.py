
import numpy as np
import json
from tqdm import tqdm
from data import RABRDataset
from rabr import RABR

# ==== Configuration ====
data_dir = "./data"
alpha = 0.1
eta = 0.1
beta = 0.5 #hyper for recall and precision
s_candidates = [5, 10, 20]
lambda_grid = np.arange(0.1, 1.01, 0.05)
embedding_path = "./item_embeddings.npy"

# ==== Load Data ====
dataset = RABRDataset(data_dir)
item_list = dataset.item_titles
item_embeddings = np.load(embedding_path, allow_pickle=True).item()  # dict {title: np.array}

# ==== Initialize Model ====
rabr = RABR(item_list, item_embeddings)
cal_data = dataset.get_calibration_prompts()

# ==== Calibration Search ====
def f_score(pred, truth, beta=1.0):
    pred_set, truth_set = set(pred), set(truth)
    tp = len(pred_set & truth_set)  #True Positives
    fp = len(pred_set - truth_set)  #False Positives
    fn = len(truth_set - pred_set)  #False Negatives
    if tp == 0:
        return 1.0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 1 - (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)

best_lambda = None
best_s = None

for s in s_candidates:
    print(f"Trying s = {s}")
    for lambda_val in lambda_grid:
        total_loss = 0
        for session_id, prompt, gt_bundle in tqdm(cal_data, desc=f"s={s}, lambda={lambda_val:.2f}"):
            completions = rabr.generate_bundles(prompt, s)
            scores = rabr.score_items(session_id, completions)
            pred_bundle = rabr.construct_bundle(scores, lambda_val)
            loss = f_score(pred_bundle, gt_bundle, beta)
            total_loss += loss

        emp_risk = total_loss / len(cal_data)
        print(f"  lambda={lambda_val:.2f}, empirical risk={emp_risk:.4f}")

        if emp_risk <= alpha:
            best_lambda = lambda_val
            best_s = s
            break  # stop at first valid lambda

    if best_lambda is not None:
        break  # stop at first s that works


def coverage_loss(pred, truth):
    truth_set = set(truth)
    if not truth_set:
        return 1.0
    hit = len(set(pred) & truth_set)
    recall = hit / len(truth_set)
    return 1 - recall  # Coverage loss = 1 - recall

best_lambda = None
best_s = None

for s in s_candidates:
    print(f"Trying s = {s}")
    for lambda_val in lambda_grid:
        total_loss = 0
        for session_id, prompt, gt_bundle in tqdm(cal_data, desc=f"s={s}, lambda={lambda_val:.2f}"):
            completions = rabr.generate_bundles(prompt, s)
            scores = rabr.score_items(session_id, completions)
            pred_bundle = rabr.construct_bundle(scores, lambda_val)
            loss = coverage_loss(pred_bundle, gt_bundle)
            total_loss += loss

        emp_risk = total_loss / len(cal_data)
        print(f"  lambda={lambda_val:.2f}, empirical risk={emp_risk:.4f}")

        if emp_risk <= alpha:
            best_lambda = lambda_val
            best_s = s
            break  # stop at first valid lambda

    if best_lambda is not None:
        break  # stop at first s that works




# ==== Save results ====
if best_lambda is not None:
    result = {"lambda_star": best_lambda, "s_star": best_s}
    with open("best_config.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Found configuration: lambda* = {best_lambda}, s* = {best_s}")
else:
    print("No valid configuration found under current alpha")
