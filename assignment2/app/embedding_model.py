
import spacy

MODEL_NAME = "en_core_web_lg"


class EmbeddingModel:
    def __init__(self):
        self.nlp = spacy.load(MODEL_NAME)

    def calculate_embedding(self, word):
        doc = self.nlp(word)
        if len(doc) != 1 or not doc[0].is_alpha:
            raise ValueError("Please enter a single alphabetic word.")
        if not doc.has_vector:
            raise LookupError(f"No pretrained vector is available for '{word}'.")
        # NumPy arrays must be converted to lists for a JSON response.
        return doc.vector.tolist()
