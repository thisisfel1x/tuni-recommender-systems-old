"""Part 1: data characterisation, temporal split, baselines (Most Popular, item-kNN) and metrics.

Usage: python run_all.py --data-dir /path/to/ml-1m
"""
import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")  # we only save figures, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

SEED = 123
K = 10  # length of the recommendation lists
REL_THRESHOLD = 4  # a held-out rating >= 4 counts as relevant (and as "liked" for item-kNN)
BATCH = 1000  # users scored at once
K_GRID = [10, 20, 50, 100, 200, 500, 1000]
SIMILARITIES = ["cosine", "jaccard"]
POPULARITY_VARIANTS = [(None, None), (None, REL_THRESHOLD), (30, None), (30, REL_THRESHOLD)]  # (window_days, min_rating)

# Step 1: load data + characterisation + plotting
def load_ratings(data_dir: Path) -> pd.DataFrame:
    # Columns: user, item, rating, timestamp
    # The file has no header and uses :: as separator
    path = Path(data_dir) / "ratings.dat"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; see README for how to obtain the data")
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["user", "item", "rating", "timestamp"],
        dtype={"user": "int32", "item": "int32", "rating": "int8", "timestamp": "int64"},
    )


def summarize(ratings: pd.DataFrame) -> dict:
    users, items, n = ratings["user"].nunique(), ratings["item"].nunique(), len(ratings)
    per_user = ratings.groupby("user").size()
    per_item = ratings.groupby("item").size().sort_values(ascending=False)
    dates = pd.to_datetime(ratings["timestamp"], unit="s")  # timestamps are unix seconds

    # share of all ratings that goes to the most popular 1% / 10% / 20% of items (long tail)
    top_share = {f"top_{int(f * 100)}%_items": float(per_item.iloc[: int(items * f)].sum() / n) for f in (0.01, 0.1, 0.2)}
    return {
        "users": users,
        "items": items,
        "interactions": n,
        "density": n / (users * items),
        "first_rating": dates.min(),
        "last_rating": dates.max(),
        "ratings_per_user_min_median_max": [int(per_user.min()), float(per_user.median()), int(per_user.max())],
        "ratings_per_item_min_median_max": [int(per_item.min()), float(per_item.median()), int(per_item.max())],
        "items_with_less_than_10_ratings": int((per_item < 10).sum()),
        "share_of_ratings_going_to": top_share,
        "rating_share": ratings["rating"].value_counts(normalize=True).sort_index().to_dict(),
    }


def make_figures(ratings: pd.DataFrame, fig_dir: Path) -> None:
    per_user = ratings.groupby("user").size().sort_values(ascending=False)
    per_item = ratings.groupby("item").size().sort_values(ascending=False)
    months = pd.to_datetime(ratings["timestamp"], unit="s").dt.to_period("M")

    def save(name):
        plt.savefig(fig_dir / name, dpi=150, bbox_inches="tight")
        plt.close()

    # long tail: rank (1 = most popular) against number of ratings, log-log
    for counts, label, name in [(per_item, "Item", "ratings_per_item.png"), (per_user, "User", "ratings_per_user.png")]:
        plt.figure()
        plt.loglog(range(1, len(counts) + 1), counts.values)
        plt.xlabel(f"{label} rank")
        plt.ylabel("Number of ratings")
        plt.title(f"Ratings per {label.lower()}")
        save(name)

    plt.figure()
    ratings["rating"].value_counts().sort_index().plot(kind="bar")
    plt.xlabel("Rating")
    plt.ylabel("Number of ratings")
    plt.title("Rating distribution")
    save("rating_distribution.png")

    plt.figure()
    months.value_counts().sort_index().plot(kind="bar")
    plt.xlabel("Month")
    plt.ylabel("Number of ratings")
    plt.title("Activity over time")
    save("activity_over_time.png")


# Step 2: temporal split
def temporal_split(ratings: pd.DataFrame, val_frac: float, test_frac: float):
    # Oldest ratings -> train, then validation, newest -> test
    val_start = ratings["timestamp"].quantile(1 - val_frac - test_frac)
    test_start = ratings["timestamp"].quantile(1 - test_frac)
    train = ratings[ratings["timestamp"] < val_start]
    val = ratings[(ratings["timestamp"] >= val_start) & (ratings["timestamp"] < test_start)]
    test = ratings[ratings["timestamp"] >= test_start]
    return train, val, test


def split_report(train, val, test) -> pd.DataFrame:
    # Size, period and how many rows come from users/items that are unknown in train (cold start)
    rows = []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        dates = pd.to_datetime(part["timestamp"], unit="s")
        rows.append(
            {
                "split": name,
                "rows": len(part),
                "from": dates.min(),
                "to": dates.max(),
                "users": part["user"].nunique(),
                "rows_with_user_in_train": part["user"].isin(train["user"]).mean(),
                "rows_with_item_in_train": part["item"].isin(train["item"]).mean(),
            }
        )
    return pd.DataFrame(rows)


# Step 3: Precision@10, Recall@10, NDCG@10
# https://towardsdatascience.com/mean-average-precision-at-k-map-k-clearly-explained-538d8e032d2/
# https://insidelearningmachines.com/precisionk_and_recallk/
def precision_at_k(ranked, relevant: set, k: int = K) -> float:
    return sum(1 for item in ranked[:k] if item in relevant) / k


def recall_at_k(ranked, relevant: set, k: int = K) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in ranked[:k] if item in relevant) / len(relevant)


# https://www.evidentlyai.com/ranking-metrics/ndcg-metric
def ndcg_at_k(ranked, relevant: set, k: int = K) -> float:
    if not relevant:
        return 0.0
    # a hit at position i (0-based) gains 1 / log2(i + 2)
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in relevant)
    # best possible: all relevant items (at most k) at the top
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal


def catalogue_coverage(all_lists, n_items: int) -> float:
    # Share of the candidate items that appear in at least one recommendation list
    return len({int(item) for ranked in all_lists for item in ranked}) / n_items


# Step 4: models
def index_data(fit: pd.DataFrame, catalogue_ids=None) -> SimpleNamespace:
    """Map ids to matrix indices (ids have gaps) and remember which items each user has seen."""
    user_ids = np.sort(fit["user"].unique())
    if catalogue_ids is None:
        item_ids = np.sort(fit["item"].unique())
    else:
        item_ids = np.sort(np.unique(catalogue_ids))
    u = np.searchsorted(user_ids, fit["user"].to_numpy())
    i = np.searchsorted(item_ids, fit["item"].to_numpy())
    shape = (len(user_ids), len(item_ids))
    seen = csr_matrix((np.ones(len(u), dtype=bool), (u, i)), shape=shape)  # any rating counts as seen
    return SimpleNamespace(
        user_ids=user_ids, item_ids=item_ids, u=u, i=i, rating=fit["rating"].to_numpy(),
        timestamp=fit["timestamp"].to_numpy(), seen=seen, shape=shape, counts=np.bincount(i, minlength=shape[1]),
    )


def most_popular(data, window_days=None, min_rating=None):
    """Score = number of ratings per item. Optionally only the last `window_days` days of the fit data
    and/or only ratings >= min_rating. The all-time count (x 1e-9) only breaks ties."""
    keep = np.ones(len(data.i), dtype=bool)
    if min_rating is not None:
        keep &= data.rating >= min_rating
    if window_days is not None:
        keep &= data.timestamp >= data.timestamp.max() - window_days * 86400
    popularity = np.bincount(data.i[keep], minlength=data.shape[1]) + 1e-9 * data.counts
    return lambda rows: np.tile(popularity, (len(rows), 1))


def popularity_name(window_days, min_rating) -> str:
    window = "all-time" if window_days is None else f"last-{window_days}d"
    counted = "all-ratings" if min_rating is None else f"rating>={min_rating}"
    return f"MostPopular({window}, {counted})"


def item_knn(data, k, similarity):
    """Item-based kNN on implicit feedback (items rated >= REL_THRESHOLD are 'liked').
    Score(user, item) = sum of the similarities between the item and the user's liked items,
    using only the k most similar neighbours of each item."""
    liked = data.rating >= REL_THRESHOLD
    profiles = csr_matrix((np.ones(liked.sum()), (data.u[liked], data.i[liked])), shape=data.shape)

    binary = profiles.astype(np.float32)
    co = (binary.T @ binary).toarray()  # co[a, b] = number of users who liked both a and b
    n = np.diag(co)  # number of users who liked each item
    if similarity == "cosine":
        sim = co / np.maximum(np.sqrt(np.outer(n, n)), 1e-12)
    elif similarity == "jaccard":
        sim = co / np.maximum(n[:, None] + n[None, :] - co, 1e-12)
    else:
        raise ValueError(f"unknown similarity: {similarity}")
    np.fill_diagonal(sim, 0)  # an item is not its own neighbour

    if k < data.shape[1]:  # keep only the k largest similarities per row
        nearest = np.argpartition(-sim, k, axis=1)[:, :k]
        keep = np.zeros(sim.shape, dtype=bool)
        np.put_along_axis(keep, nearest, True, axis=1)
        sim = np.where(keep, sim, 0)
    sim_t = csr_matrix(sim.T.astype(np.float64))  # row a of sim = neighbours of a  ->  score = profiles @ sim.T

    return lambda rows: (profiles[rows] @ sim_t).toarray() + 1e-9 * data.counts  # popularity breaks ties


# step 5: evaluation
def evaluate(score, data, eval_ratings: pd.DataFrame) -> dict:
    """Average metrics over users who have >= 1 relevant item in `eval_ratings` and a history in the fit data.
    Items a user saw during fitting are excluded; ties are broken by item order, so results are deterministic."""
    relevant = eval_ratings[eval_ratings["rating"] >= REL_THRESHOLD].groupby("user")["item"].agg(set)
    known = np.isin(relevant.index.to_numpy(), data.user_ids)
    users = relevant.index.to_numpy()[known]
    rows = np.searchsorted(data.user_ids, users)

    precision, recall, ndcg, lists = [], [], [], []
    for start in range(0, len(users), BATCH):
        batch = rows[start : start + BATCH]
        scores = np.where(data.seen[batch].toarray(), -np.inf, score(batch))
        top = np.argsort(-scores, axis=1, kind="stable")[:, :K]
        for user, ranked in zip(users[start : start + BATCH], data.item_ids[top]):
            truth = relevant[user]
            precision.append(precision_at_k(ranked, truth))
            recall.append(recall_at_k(ranked, truth))
            ndcg.append(ndcg_at_k(ranked, truth))
            lists.append(ranked)
    return {
        f"precision@{K}": float(np.mean(precision)),
        f"recall@{K}": float(np.mean(recall)),
        f"ndcg@{K}": float(np.mean(ndcg)),
        "coverage": catalogue_coverage(lists, len(data.item_ids)),
        "n_users": len(users),
        "n_cold_users_skipped": int((~known).sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"), required=True, help="directory containing ratings.dat")
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--fig-dir", type=Path, default=Path("figures"))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--val-frac", type=float, default=0.05, help="share of the newest ratings used for validation")
    p.add_argument("--test-frac", type=float, default=0.05, help="share of the newest ratings used for test")
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    # data
    ratings = load_ratings(args.data_dir)
    movies = pd.read_csv(
        args.data_dir / "movies.dat",
        sep="::",
        engine="python",
        header=None,
        names=["item", "title", "genres"],
        encoding="latin-1",
    )

    catalogue_ids = movies["item"].to_numpy()
    summary = summarize(ratings)
    (args.out_dir / "data_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    make_figures(ratings, args.fig_dir)
    print(json.dumps(summary, indent=2, default=str))

    # split
    train, val, test = temporal_split(ratings, args.val_frac, args.test_frac)
    report = split_report(train, val, test)
    report.to_csv(args.out_dir / "split_report.csv", index=False)
    print("\n" + report.to_string(index=False))

    # tuning on validation only: fit on train, evaluate on val
    train_data = index_data(train, catalogue_ids)
    tuning = []
    for window, min_rating in POPULARITY_VARIANTS:
        metrics = evaluate(most_popular(train_data, window, min_rating), train_data, val)
        tuning.append({"model": popularity_name(window, min_rating), "split": "val", **metrics})
    for similarity in SIMILARITIES:
        for k in K_GRID:
            metrics = evaluate(item_knn(train_data, k, similarity), train_data, val)
            tuning.append({"model": f"ItemKNN(k={k}, {similarity})", "split": "val", "k": k, "similarity": similarity, **metrics})
            print(f"val  ItemKNN(k={k}, {similarity}): ndcg@{K}={metrics[f'ndcg@{K}']:.4f}")
    pd.DataFrame(tuning).to_csv(args.out_dir / "tuning_val.csv", index=False)

    best = max((row for row in tuning if row["model"].startswith("ItemKNN")), key=lambda row: row[f"ndcg@{K}"])
    print(f"\nBest ItemKNN on validation: k={best['k']}, similarity={best['similarity']}")

    # final: refit on train + val with the chosen settings, evaluate once on test
    full_data = index_data(pd.concat([train, val]), catalogue_ids)
    models = {popularity_name(w, m): most_popular(full_data, w, m) for w, m in POPULARITY_VARIANTS}
    models[f"ItemKNN(k={best['k']}, {best['similarity']})"] = item_knn(full_data, best["k"], best["similarity"])
    results = []
    for name, score in models.items():
        metrics = evaluate(score, full_data, test)
        results.append({"model": name, "split": "test", "seed": args.seed, **metrics})
        print(f"test {name}: " + ", ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in metrics.items()))
    pd.DataFrame(results).to_csv(args.out_dir / "results.csv", index=False)


if __name__ == "__main__":
    main()