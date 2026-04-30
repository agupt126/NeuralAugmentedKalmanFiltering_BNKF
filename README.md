# Neural-Augmented Kalman Filtering (BNKF)

## Author
Akhil Gupta | JHUEP Fall 2024 | agupt126@jh.edu

## Overview
This repository implements nonlinear state estimation for UAV trajectories using physics-informed Bayesian neural filtering and Kalman filter baselines. It includes:

- `models/bnkf.py`: Bayesian nonlinear Kalman filter model
- `models/bnn.py` and `models/bnn_ensemble.py`: neural baseline models
- `models/stonesoup_radar_sim.py`: Stone Soup EKF/UKF comparison utilities
- `helper/`: data loading, preprocessing, evaluation, and visualization utilities
- `notebooks/`: analysis and comparison notebooks
- `tests/test_evaluator.py`: evaluator unit test

## Repository layout

- `dataset/Synthetic-UAV-Flight-Trajectories/`: raw trajectory CSV files
- `dataset/dataframe-readins/`: generated train/test split data
- `models/saved_weights/`: pretrained weights for experimentation
- `models/deprecated/`: legacy implementations and earlier experiments

## Setup

1. Clone the repository.
2. Create a Python environment with Python 3.10 or newer.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the dataset from:

https://huggingface.co/datasets/riotu-lab/Synthetic-UAV-Flight-Trajectories

5. Place all trajectory CSV files in:

```text
dataset/Synthetic-UAV-Flight-Trajectories/
```

6. If necessary, adjust the dataset path in `helper/training_preprocess.py` before running preprocessing or evaluation.

## Usage

- Preprocess raw data and build train/test splits:

```bash
python helper/training_preprocess.py
```

- Run evaluation and visualization helpers:

```bash
python helper/eval_worker.py
```

- Run the evaluator unit test:

```bash
python -m unittest tests/test_evaluator.py
```

- Explore notebooks in:

  - `notebooks/comparisons/`
  - `notebooks/report/`
  - `notebooks/testing/`

## Notes

- The primary experimental code is in `models/`.
- `helper/data_handler.py` reads trajectory CSVs from the dataset folder and can export visualizations as GIFs using `pillow`.
- Pretrained weights are available in `models/saved_weights/`.

## Resources

- Dataset: https://huggingface.co/datasets/riotu-lab/Synthetic-UAV-Flight-Trajectories
- Stone Soup: https://github.com/Adelson-Velsky/Stone-Soup
- Physics-informed Bayesian neural filtering implementation: `models/bnkf.py`
