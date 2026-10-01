"""A small bigram implementation for the Module 3 classroom API."""

import random
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self, corpus):
        self.transitions = defaultdict(Counter)
        for text in corpus:
            words = text.lower().split()
            for current, following in zip(words, words[1:]):
                self.transitions[current][following] += 1

    def generate_text(self, start_word, length):
        words = [start_word.lower()]
        for _ in range(length - 1):
            choices = self.transitions.get(words[-1])
            if not choices:
                break
            words.append(random.choices(list(choices), weights=list(choices.values()))[0])
        return " ".join(words)
