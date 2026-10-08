def accuracy(y_true, y_pred):
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def counts(y_true, y_pred, label):
    """True positives, false positives and false negatives for one label."""
    tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
    fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
    fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
    return tp, fp, fn


def precision(y_true, y_pred, label):
    """Of the reviews we labelled `label`, how many really were."""
    tp, fp, fn = counts(y_true, y_pred, label)
    return tp / (tp + fp) if tp + fp else 0.0


def recall(y_true, y_pred, label):
    """Of the reviews that really were `label`, how many we found."""
    tp, fp, fn = counts(y_true, y_pred, label)
    return tp / (tp + fn) if tp + fn else 0.0


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def confusion_matrix(y_true, y_pred, labels):
    """matrix[actual][predicted] = how many reviews."""
    matrix = {a: {p: 0 for p in labels} for a in labels}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix
