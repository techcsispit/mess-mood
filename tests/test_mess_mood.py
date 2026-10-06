from mess_mood.data import load_reviews
from mess_mood.evaluate import fit, k_fold
from mess_mood.metrics import accuracy, confusion_matrix, f1, recall
from mess_mood.text import tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The DOSA was crispy") == ["dosa", "crispy", "dosa_crispy"]


def test_never_is_joined_to_the_next_word():
    assert tokenize("never fresh food") == ["not_fresh", "food", "not_fresh_food"]


def test_dataset():
    rows = load_reviews()
    assert len(rows) >= 100
    assert {label for _, label in rows} == {"positive", "negative"}


def test_obvious_reviews():
    model = fit(load_reviews())
    assert model.predict("delicious and tasty") == "positive"
    assert model.predict("cold and stale") == "negative"
    assert 0.5 <= model.confidence("delicious") <= 1


def test_unknown_words_are_ignored():
    assert fit(load_reviews()).predict("xyzzy qwerty") in {"positive", "negative"}


def test_metrics():
    y_true = ["pos", "pos", "neg", "neg"]
    y_pred = ["pos", "neg", "neg", "neg"]
    assert accuracy(y_true, y_pred) == 0.75
    assert recall(y_true, y_pred, "pos") == 0.5
    assert f1(1.0, 0.5) == 2 / 3
    assert confusion_matrix(y_true, y_pred, ["neg", "pos"]) == {"neg": {"neg": 2, "pos": 0}, "pos": {"neg": 1, "pos": 1}}


def test_k_fold_test_sets_dont_overlap():
    rows = [(str(i), "x") for i in range(20)]
    tests = [set(test) for _, test in k_fold(rows, k=5)]
    assert len(tests) == 5
    assert all(not (a & b) for i, a in enumerate(tests) for b in tests[i + 1:])


def test_model_save_and_load(tmp_path):
    from mess_mood.model import NaiveBayes
    
    model = fit(load_reviews())
    path = tmp_path / "model.json"
    model.save(path)
    
    loaded_model = NaiveBayes.load(path)
    test_text = "the food was not fresh but it was okay"
    assert model.predict(test_text) == loaded_model.predict(test_text)
    assert model.scores(test_text) == loaded_model.scores(test_text)
