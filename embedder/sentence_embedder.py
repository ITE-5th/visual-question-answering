import torch
from torchwordemb import load_word2vec_bin

from embedder.embedder import Embedder


class SentenceEmbedder(Embedder):
    def __init__(self, word2vec_bin_file_path: str):
        """
        create a sentence embedder given word2vec pretrained model path
        :param word2vec_bin_file_path: the word2vec pretrained file path
        """
        self.vocab, self.tensor = load_word2vec_bin(word2vec_bin_file_path)

    def embed(self, sentence: str) -> torch.FloatTensor:
        """
        embed a sentence using word2vec model given in constructor
        :param sentence: the sentence to embed
        :return: a tensor whose rows contain the embedding of the words
        """
        return self.tensor[[self.vocab[word] for word in sentence.split(" ")], :]
