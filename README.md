# RABR

This repository provides a reference implementation of **RABR** for bundle recommendation.

---

## Requirements

- Python 3.8+
- Required packages:
  - numpy
  - tqdm
  - scikit-learn
  - sentence-transformers
  - openai (or compatible LLM API)

Please install dependencies in your own environment.

---

## Files

```text
RABR/
├── rabr.py             # Core algorithm
├── data.py             # Dataset loading
├── evaluation.py       # Evaluation metrics
├── run_calibrate.py    # Calibration script (find λ*)
├── run_inference.py    # Inference script (generate bundles)
└── README.md
```

---

## Usage

### 1. Prepare Dataset

Modify `data.py` to load your dataset (Electronics / Clothing / Food).

---

### 2. Calibration

Run calibration to find the optimal threshold λ*:

```bash
python run_calibrate.py
```

This will save the calibrated threshold for inference.

---

### 3. Inference

Generate bundles on the test set using the calibrated threshold:

```bash
python run_inference.py
```

---

### 4. Evaluation

Evaluate the generated bundles:

```bash
python evaluation.py
```




