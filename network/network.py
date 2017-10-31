import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adadelta

from embedder.sentence_embedder import SentenceEmbedder


class Network(nn.Module):
    def __init__(self, max_sentence_length: int, word_vector_length: int = 300):
        super().__init__()
        self.max_sentence_length = max_sentence_length
        self.gru = nn.GRU(word_vector_length, 512)

    def forward(self, x):
        words_vec = x[:self.max_sentence_length, :]
        inputs = words_vec.view(self.max_sentence_length, 1, words_vec.data.shape[1])
        hidden = autograd.Variable(torch.randn(1, 1, 512))
        out, hidden = self.gru(inputs, hidden)
        return hidden


if __name__ == '__main__':
    embedder = SentenceEmbedder("data/GoogleNews-vectors-negative300.bin")
    temp = "small test"
    net = Network(5, embedder.max_sentence_length)
    temp = embedder.embed(temp)
    x = Variable(temp, requires_grad=False)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adadelta(net.parameters(), lr=0.15)
    epochs = 20
    for epoch in range(epochs):
        loss = loss_function(None, None)
        loss.backward()
        optimizer.step()
