import unittest
import torch

from embedder.sentence_embedder import SentenceEmbedder, EmbeddingType


class SentenceEmbedderTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.max_sentence_length = 14
        self.embedder = SentenceEmbedder("../../data/GoogleNews-vectors-negative300.bin", EmbeddingType.WORD2VEC, self.max_sentence_length)

    def test_embed_sentence(self):
        temp = "small test"
        temp = self.embedder.embed(temp)
        self.assertEqual(temp.shape, (self.max_sentence_length, 300))


if __name__ == '__main__':
    unittest.main()