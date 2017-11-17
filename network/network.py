import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adadelta
from network.gated_hyperbolic_tangent import GatedHyperbolicTangent


class Network(nn.Module):
    def __init__(self, max_sentence_length: int, word_vector_length: int = 300, image_location_number : int = 36):
        super().__init__()
        self.max_sentence_length = max_sentence_length
        self.gru = nn.GRU(word_vector_length, 512)
        # image embedding weights
        self.image_a = Parameter(torch.randn(image_location_number).view(image_location_number, 1))
        # image GatedHyperbolicTangent
        self.ght1 = GatedHyperbolicTangent(512 + 2048, image_location_number)
        # question GatedHyperbolicTangent
        self.ght2 = GatedHyperbolicTangent(512, 512)
        # image attentions GatedHyperbolicTangent
        self.ght3 = GatedHyperbolicTangent(2048, 512)
        # output GatedHyperbolicTangent
        self.ght4 = GatedHyperbolicTangent(512, 512)
        self.output_weights = Parameter(torch.randn(1, 512))

    def forward(self, x):
        words_vec = x[:self.max_sentence_length, :]
        inputs = words_vec.view(self.max_sentence_length, 1, words_vec.data.shape[1])
        hidden = autograd.Variable(torch.randn(1, 1, 512))
        out, hidden = self.gru(inputs, hidden)
        question_embedding = hidden.view(1, 512)
        image_features = x[self.max_sentence_length + 1:, :]
        question_embedding_mat = question_embedding.repeat(image_features.shape[0], 1)
        fusion_features = torch.cat([image_features, question_embedding_mat], 1)
        fusion_features = self.ght1(fusion_features)
        fusion_features = fusion_features * self.image_a
        attentions = F.softmax(fusion_features)
        v = torch.sum(image_features * attentions, 0, keepdim=True)
        h = self.ght2(question_embedding.view(512, 1)) * self.ght3(v)
        s = F.sigmoid(torch.mm(self.output_weights, h))
        return hidden


if __name__ == '__main__':
    net = Network(14)
    net = nn.DataParallel(net).cuda()
    optimizer = Adadelta(net.parameters())
    criterion = nn.CrossEntropyLoss()
    epochs = 20
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs, labels = None, None
        loss = criterion(outputs, labels)
        loss.backward()
