
import numpy as np
import json
from tqdm import tqdm
from src.data import RABRDataset
from src.rabr import RABR

# ==== Load calibrated config ====
with open("best_config.json", "r") as f:
    config = json.load(f)

lambda_star = config["lambda_star"]
s_star = config["s_star"]

# ==== Paths ====
data_dir = "./data"
embedding_path = "./item_embeddings.npy"
output_path = "output_predictions.json"

# ==== Load Data ====
dataset = RABRDataset(data_dir)
item_list = dataset.item_titles
item_embeddings = np.load(embedding_path, allow_pickle=True).item()
test_data = dataset.get_test_prompts()

# ==== Initialize RABR Model ====
rabr = RABR(item_list, item_embeddings)

# ==== Inference ====
results = []
for session_id, prompt, gt_bundle in tqdm(test_data, desc="Inference"):
    completions = rabr.generate_bundles(prompt, s_star)
    scores = rabr.score_items(session_id, completions)
    pred_bundle = rabr.construct_bundle(scores, lambda_star)

    results.append({
        "session_id": session_id,
        "prompt": prompt,
        "predicted_bundle": pred_bundle,
        "ground_truth_bundle": gt_bundle
    })

# ==== Save output ====
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Inference completed. Results saved to {output_path}")
