import string
from enum import Enum

import nltk
import torch
from torchwordemb import load_word2vec_bin, load_glove_text

from embedder.embedder import Embedder


class EmbeddingType(Enum):
    WORD2VEC = 1
    GLOVE = 2


class SentenceEmbedder(Embedder):
    def __init__(self, file_path: str, embedding_type: EmbeddingType = EmbeddingType.WORD2VEC, max_sentence_length: int = 14):
        """
        create a sentence embedder given word2vec pretrained model path
        :param word2vec_bin_file_path: the word2vec pretrained file path
        """
        self.max_sentence_length = max_sentence_length
        self.vocab, self.tensor = load_word2vec_bin(
            file_path) if embedding_type == EmbeddingType.WORD2VEC else load_glove_text(file_path)

    def embed(self, sentence: str) -> torch.FloatTensor:
        """
        embed a sentence using word2vec model given in constructor
        :param sentence: the sentence to embed
        :return: a tensor whose rows contain the embedding of the words
        """
        indices = [self.vocab[word] for word in nltk.word_tokenize(sentence) if word not in string.punctuation]
        l = len(indices)
        if l > self.max_sentence_length:
            raise ValueError("Your sentence length is larger then max sentence length")
        temp = self.tensor[indices, :]
        if l == self.max_sentence_length:
            return temp
        else:
            padding_size = self.max_sentence_length - l
            t = torch.zeros(padding_size, 300)
            return torch.cat((temp, t))
