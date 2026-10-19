"""Naive Bayes, written out by hand.

To score a review for a label:

    log P(label) + sum of log P(word | label) for each word in the review

and pick the label with the higher score. P(word | label) uses add-one
smoothing, so a word never seen with a label gets a small probability
instead of zero. Words the model has never seen at all are ignored.
"""
import json
import math
from collections import Counter

from mess_mood.text import TOKENIZER_VERSION, tokenize


class NaiveBayes:
    def fit(self, texts, labels):
        texts = list(texts)
        labels = list(labels)
        if len(texts) != len(labels):
            raise ValueError(
                "texts and labels must contain the same number of items "
                f"(got {len(texts)} texts and {len(labels)} labels)"
            )

        self.label_counts = Counter(labels)
        self.word_counts = {label: Counter() for label in self.label_counts}
        for text, label in zip(texts, labels):
            self.word_counts[label].update(tokenize(text))
        self.vocab = set().union(*self.word_counts.values())
        self.totals = {label: sum(c.values()) for label, c in self.word_counts.items()}
        return self

    def word_prob(self, word, label):
        return (self.word_counts[label][word] + 1) / (self.totals[label] + len(self.vocab))

    def scores(self, text):
        n = sum(self.label_counts.values())
        words = [w for w in tokenize(text) if w in self.vocab]
        return {
            label: math.log(count / n) + sum(math.log(self.word_prob(w, label)) for w in words)
            for label, count in self.label_counts.items()
        }

    def predict(self, text):
        scores = self.scores(text)
        return max(scores, key=scores.get)

    def confidence(self, text):
        scores = self.scores(text)
        best = max(scores.values())
        weights = [math.exp(s - best) for s in scores.values()]
        return max(weights) / sum(weights)

    def top_words(self, label, n=10):
        """Words that point most strongly towards `label`."""
        others = [l for l in self.label_counts if l != label]

        def ratio(word):
            other = sum(self.word_prob(word, o) for o in others) / len(others)
            return self.word_prob(word, label) / other

        return sorted(self.vocab, key=ratio, reverse=True)[:n]

    def save(self, path):
        data = {
            "tokenizer_version": TOKENIZER_VERSION,
            "label_counts": dict(self.label_counts),
            "word_counts": {label: dict(counts) for label, counts in self.word_counts.items()},
            "vocab": list(self.vocab),
            "totals": self.totals
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)

        saved_version = data.get("tokenizer_version")
        if saved_version is None:
            raise ValueError(
                "Saved model has no tokenizer version and cannot be loaded safely; "
                "retrain and save the model with the current version."
            )
        if saved_version != TOKENIZER_VERSION:
            raise ValueError(
                f"Saved model uses tokenizer version {saved_version}, but this "
                f"installation uses version {TOKENIZER_VERSION}; retrain the model."
            )
        
        model = cls()
        model.label_counts = Counter(data["label_counts"])
        model.word_counts = {label: Counter(counts) for label, counts in data["word_counts"].items()}
        model.vocab = set(data["vocab"])
        model.totals = data["totals"]
        return model
