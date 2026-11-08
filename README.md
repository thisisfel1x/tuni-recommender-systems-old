# Recommender Systems Course Project, Group 15

Charleen Lang, Felix Weber, Vinicius Martins

## Setup

Python 3.11, dependencies pinned in `requirements.txt`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download MovieLens 1M from https://grouplens.org/datasets/movielens/1m/ and unzip it.
The data directory must contain `ratings.dat` and `movies.dat`.

## Run

```bash
python run_all.py --data-dir /path/to/ml-1m
```

This reproduces all numbers and figures of the report (about 20 s):

- `results/results.csv`: final test results, same columns as the course results template
- `results/tuning_val.csv`: all validation runs used for tuning
- `results/split_report.csv`, `results/data_summary.json`: split and dataset statistics
- `figures/`: plots used in the report

The seed is fixed to 123 (`SEED` in `run_all.py`, can be changed with `--seed`).
Tests: `python -m pytest`
