import random

from mess_mood.data import load_reviews
from mess_mood.metrics import accuracy, confusion_matrix, f1, precision, recall
from mess_mood.model import NaiveBayes


def train_test_split(rows, test_size=0.2, seed=42):
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    cut = int(len(rows) * (1 - test_size))
    return rows[:cut], rows[cut:]


def k_fold(rows, k=5, seed=42):
    """Yields (train, test) k times. Every row is in exactly one test set."""
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    fold_size = len(rows) // k
    for i in range(k):
        test = rows[i * fold_size:(i + 1) * fold_size]
        train = rows[:i * fold_size] + rows[(i + 1) * fold_size:]
        yield train, test


def fit(rows):
    return NaiveBayes().fit([t for t, _ in rows], [l for _, l in rows])


def report(rows=None):
    rows = rows or load_reviews()
    train, test = train_test_split(rows)
    model = fit(train)
    y_true = [l for _, l in test]
    y_pred = [model.predict(t) for t, _ in test]
    labels = sorted(set(y_true) | set(y_pred))

    print(f"Trained on {len(train)} reviews, tested on {len(test)}.\n")
    print(f"Accuracy: {accuracy(y_true, y_pred):.0%}\n")
    print(f"{'':10} precision  recall     f1")
    for label in labels:
        p, r = precision(y_true, y_pred, label), recall(y_true, y_pred, label)
        print(f"{label:10} {p:9.2f} {r:7.2f} {f1(p, r):6.2f}")

    matrix = confusion_matrix(y_true, y_pred, labels)
    print("\nConfusion matrix (rows: actual, columns: predicted)")
    print(" " * 10 + "".join(f"{l:>10}" for l in labels))
    for actual in labels:
        print(f"{actual:10}" + "".join(f"{matrix[actual][p]:10}" for p in labels))

    scores = [accuracy([l for _, l in te], [fit(tr).predict(t) for t, _ in te]) for tr, te in k_fold(rows)]
    print(f"\n5-fold cross-validation accuracy: {sum(scores) / len(scores):.0%}")
