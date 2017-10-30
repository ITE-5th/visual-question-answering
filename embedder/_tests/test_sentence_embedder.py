import unittest
import torch

from embedder.sentence_embedder import SentenceEmbedder


class SentenceEmbedderTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.embedder = SentenceEmbedder("data/GoogleNews-vectors-negative300.bin")

    def test_embed_sentence(self):
        temp = "small test"
        # need to fill the tensor
        tens_a, tens_b = self.embedder.embed(temp), torch.FloatTensor()
        self.assertTrue(torch.all(torch.lt(torch.abs(torch.add(tens_a, -tens_b)), 1e-12)))