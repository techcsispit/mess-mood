"""Tests for the commands in the README's "Running it" section, `confidence`,
`top_words` and what `report()` prints."""
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from mess_mood.data import load_reviews
from mess_mood.evaluate import fit, report, train_test_split

ROOT = Path(__file__).parent.parent

SMALL = [
    ("great tasty food", "positive"),
    ("tasty fresh meal", "positive"),
    ("fresh great lunch", "positive"),
    ("great lunch", "positive"),
    ("bland soggy food", "negative"),
    ("soggy stale meal", "negative"),
    ("stale bland lunch", "negative"),
    ("bland lunch", "negative"),
]


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "mess_mood", *args],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )


# predict

@pytest.mark.parametrize("review, label", [
    ("the dosa was cold and soggy", "negative"),  # the README's own example
    ("fresh and tasty", "positive"),
])
def test_predict_prints_label_and_confidence(review, label):
    result = run_cli("predict", review)
    assert result.returncode == 0
    assert re.fullmatch(rf"{label} \((\d+)% sure\)\n", result.stdout)


def test_predict_without_a_review_is_a_usage_error():
    assert run_cli("predict").returncode == 2


# confidence

def test_confidence_is_the_softmax_of_the_scores():
    model = fit(SMALL)
    scores = model.scores("great tasty")
    expected = max(math.exp(s) for s in scores.values()) / sum(math.exp(s) for s in scores.values())
    assert model.confidence("great tasty") == pytest.approx(expected)
    assert model.confidence("great tasty") > 0.9


def test_confidence_is_half_when_every_word_is_unknown():
    # unknown words are ignored and the two labels are equally common
    assert fit(SMALL).confidence("xyzzy plugh") == pytest.approx(0.5)


# top-words

def test_top_words_points_towards_the_label_strongest_first():
    # one-word reviews, so there are no word pairs, and distinct counts, so no ties
    reviews = [("great", "positive")] * 3 + [("tasty", "positive")] * 2 + [("fresh", "positive")]
    reviews += [("bland", "negative")] * 3 + [("soggy", "negative")] * 2 + [("stale", "negative")]
    model = fit(reviews)
    assert model.top_words("positive", 3) == ["great", "tasty", "fresh"]
    assert model.top_words("negative", 3) == ["bland", "soggy", "stale"]


def test_top_words_returns_n_words_and_n_defaults_to_ten():
    model = fit(SMALL)
    assert len(model.top_words("positive", 4)) == 4
    assert len(model.top_words("positive")) == 10


def top_words_output(*args):
    """Run `top-words` and return {label: [words]}. Words with equal scores can
    come out in any order (the vocabulary is a set), so the tests below check
    which words each line holds, not their exact order."""
    result = run_cli("top-words", *args)
    assert result.returncode == 0
    lines = result.stdout.splitlines()
    assert [line.split(": ", 1)[0] for line in lines] == ["negative", "positive"]
    return {line.split(": ", 1)[0]: line.split(": ", 1)[1].split(", ") for line in lines}


def test_top_words_command_lists_ten_words_that_point_towards_each_label():
    model = fit(load_reviews())
    shown = top_words_output()
    for label, other in [("negative", "positive"), ("positive", "negative")]:
        assert len(shown[label]) == 10
        assert len(set(shown[label])) == 10
        for word in shown[label]:
            assert model.word_prob(word, label) > model.word_prob(word, other)


def test_top_words_command_n_sets_how_many_words_each_label_shows():
    shown = top_words_output("-n", "3")
    assert [len(words) for words in shown.values()] == [3, 3]


# evaluate

def test_evaluate_command_prints_the_report():
    result = run_cli("evaluate")
    assert result.returncode == 0
    assert result.stdout.startswith("Trained on ")
    assert "Confusion matrix (rows: actual, columns: predicted)" in result.stdout
    assert re.search(r"5-fold cross-validation accuracy: \d+%\n$", result.stdout)


def test_report_prints_split_sizes_accuracy_per_label_scores_and_matrix(capsys):
    rows = [(f"great tasty fresh {i}", "positive") for i in range(25)]
    rows += [(f"bland soggy stale {i}", "negative") for i in range(25)]
    # mislabelled on purpose, so the model gets one test review wrong
    rows += [(f"great tasty fresh odd {i}", "negative") for i in range(6)]
    train, test = train_test_split(rows)
    model = fit(train)
    seen = Counter((label, model.predict(text)) for text, label in test)
    assert seen[("negative", "positive")] == 1 and seen[("positive", "negative")] == 0
    correct = seen[("negative", "negative")] + seen[("positive", "positive")]

    report(rows)
    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == f"Trained on {len(train)} reviews, tested on {len(test)}."
    assert lines[2] == f"Accuracy: {correct / len(test):.0%}"
    assert lines[4].split() == ["precision", "recall", "f1"]
    for line, label in [(lines[5], "negative"), (lines[6], "positive")]:
        assert re.fullmatch(rf"{label} +\d\.\d\d +\d\.\d\d +\d\.\d\d", line)

    assert lines[8] == "Confusion matrix (rows: actual, columns: predicted)"
    assert lines[9].split() == ["negative", "positive"]
    assert lines[10].split() == ["negative", str(seen[("negative", "negative")]), "1"]
    assert lines[11].split() == ["positive", "0", str(seen[("positive", "positive")])]

    assert re.fullmatch(r"5-fold cross-validation accuracy: \d+%", lines[13])
