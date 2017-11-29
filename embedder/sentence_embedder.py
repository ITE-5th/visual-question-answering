import nltk
import torch

from embedder.embedder import Embedder


class SentenceEmbedder(Embedder):
    def __init__(self, vocabs, sentence_length=14):
        self.vocabs = vocabs
        self.sentence_length = sentence_length

    def embed(self, sentence: str) -> torch.LongTensor:
        words = nltk.word_tokenize(sentence)
        if len(words) > self.sentence_length:
            words = words[:self.sentence_length]
        indices = [0] * self.sentence_length
        for i in range(len(words)):
            try:
                indices[i] = self.vocabs[words[i]]
            except:
                indices[i] = 0
        return torch.LongTensor(indices)
