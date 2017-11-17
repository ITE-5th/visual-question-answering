import string
from enum import Enum

import nltk
import torch
from gensim.models import KeyedVectors
from torchwordemb import load_word2vec_bin, load_glove_text

from embedder.embedder import Embedder


class EmbeddingType(Enum):
    WORD2VEC = 1
    GLOVE = 2
    FASTTEXT = 3


class SentenceEmbedder(Embedder):
    def __init__(self, file_path: str, embedding_type: EmbeddingType = EmbeddingType.FASTTEXT,
                 max_sentence_length: int = 14):
        """
        :param file_path: pretrained model file path
        :param embedding_type: the file embedding type
        :param max_sentence_length:
        """
        self.max_sentence_length = max_sentence_length
        self.embedding_type = embedding_type
        if embedding_type == EmbeddingType.GLOVE or embedding_type == EmbeddingType.WORD2VEC:
            self.vocab, self.tensor = load_word2vec_bin(
                file_path) if embedding_type == EmbeddingType.WORD2VEC else load_glove_text(file_path)
        else:
            self.fast_text = KeyedVectors.load(file_path)

    def embed(self, sentence: str) -> torch.FloatTensor:
        """
        embed a sentence using word2vec model given in constructor
        :param sentence: the sentence to embed
        :return: a tensor whose rows contain the embedding of the words
        """
        words = [word for word in nltk.word_tokenize(sentence) if word not in string.punctuation]
        l = len(words)
        if l > self.max_sentence_length:
            words = words[:self.max_sentence_length]
            l = self.max_sentence_length
        if self.embedding_type == EmbeddingType.GLOVE or self.embedding_type == EmbeddingType.WORD2VEC:
            indices = [self.vocab[word] for word in words]
            temp = self.tensor[indices, :]
        else:
            temp = torch.from_numpy(self.fast_text[[word for word in words]])
        if l == self.max_sentence_length:
            return temp
        else:
            padding_size = self.max_sentence_length - l
            t = torch.zeros(padding_size, 300)
            return torch.cat((temp, t))
