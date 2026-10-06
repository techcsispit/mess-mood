STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "was", "were", "are", "be", "been",
    "i", "me", "my", "we", "it", "its", "this", "that", "to", "of", "in", "on", "at",
    "for", "with", "so", "very", "just", "again", "today", "no", "not",
}

NEGATIONS = {"not", "no", "never"}


def tokenize(text):
    """Turns a review into a list of words for the model.

    Stopwords are dropped. A negation is joined to the word after it,
    so "not fresh" becomes "not_fresh".
    """
    words = [w for w in text.lower().split() if w not in STOPWORDS]
    tokens = []
    negate = False
    for word in words:
        if word in NEGATIONS:
            negate = True
            continue
        tokens.append(f"not_{word}" if negate else word)
        negate = False
    
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams
