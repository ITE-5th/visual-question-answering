import torch
from torchwordemb import load_word2vec_bin

from embedder.embedder import Embedder


class SentenceEmbedder(Embedder):
    def __init__(self, word2vec_bin_file_path: str):
        self.vocab, self.tensor = load_word2vec_bin(word2vec_bin_file_path)

    def embed(self, sentence: str) -> torch.FloatTensor:
        return self.tensor[[self.vocab[word] for word in sentence.split(" ")], :]
