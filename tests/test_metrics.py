"""Unit tests for recommender evaluation metrics."""

import numpy as np
import pandas as pd

from run_all import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    catalogue_coverage,
    temporal_split,
    index_data,
)


def test_precision_at_k():
    ranked = [1, 2, 3, 4, 5]
    relevant = {1, 3}

    assert precision_at_k(ranked, relevant, k=5) == 0.4


def test_recall_at_k():
    ranked = [1, 2, 3, 4, 5]
    relevant = {1, 3, 6, 7}

    assert recall_at_k(ranked, relevant, k=5) == 0.5


def test_ndcg_perfect_ranking():
    ranked = [1, 2, 3]
    relevant = {1, 2, 3}

    assert np.isclose(ndcg_at_k(ranked, relevant, k=3), 1.0)


def test_ndcg_empty_relevance():
    assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


def test_catalogue_coverage():
    recommendations = [[1, 2], [2, 3]]

    assert catalogue_coverage(recommendations, 5) == 0.6


def test_temporal_split():
    ratings = pd.DataFrame({
        "user": [1] * 10,
        "item": list(range(10)),
        "rating": [4] * 10,
        "timestamp": list(range(1, 11)),
    })

    train, val, test = temporal_split(
        ratings, val_frac=0.2, test_frac=0.2
    )

    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2

    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()

    assert len(train) + len(val) + len(test) == len(ratings)


def test_full_catalogue():
    ratings = pd.DataFrame({
        "user": [1, 2],
        "item": [10, 20],
        "rating": [5, 4],
        "timestamp": [100, 200],
    })

    catalogue = [10, 20, 30]

    data = index_data(ratings, catalogue)

    assert len(data.item_ids) == 3
    assert set(data.item_ids) == {10, 20, 30}
    assert data.shape == (2, 3)

    assert data.counts[2] == 0
