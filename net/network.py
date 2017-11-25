import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from torch.autograd import Variable
from torch.nn import Parameter, MultiLabelSoftMarginLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.VqaDataset import VqaDataset
from net.gated_hyperbolic_tangent import GatedHyperbolicTangent


class Network(nn.Module):
    def __init__(self, question_vocab_size: int, answer_vocab_size: int, initial_embedding_weights=None,
                 word_vector_length: int = 300,
                 image_location_number: int = 36):
        super().__init__()
        self.embedding = nn.Embedding(question_vocab_size, word_vector_length)
        if initial_embedding_weights is not None:
            self.embedding.weight.data.copy_(initial_embedding_weights)
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
        self.output = nn.Linear(512, answer_vocab_size)

    def forward(self, question, image_features):
        question_vectors = self.embedding(question)
        print("question vectors shape = " + str(question_vectors.data.shape))
        words_length = question_vectors.data.shape[0]
        inputs = question_vectors.view(words_length, 1, question_vectors.data.shape[1])
        hidden = autograd.Variable(torch.randn(1, 1, 512))
        out, hidden = self.gru(inputs, hidden)
        question_embedding = hidden.view(-1, 512)
        question_embedding_mat = question_embedding.repeat(1, image_features.size(1)).view(-1, image_features.size(1),
                                                                                           512)
        print("question embedding shape = " + str(question_embedding_mat.data.shape))
        image_features = F.normalize(image_features, -1)
        fusion_features = torch.cat((image_features, question_embedding_mat), -1)
        fusion_features = self.ght1(fusion_features)
        print("fusion feature 1 =" + str(fusion_features.data.shape))
        fusion_features = fusion_features * self.image_a
        attentions = F.softmax(fusion_features)
        v = torch.sum(image_features * attentions, 0, keepdim=True)
        print("v = " + str(v.data.shape))
        h = self.ght2(question_embedding.view(512, 1)) * self.ght3(v)
        h = self.ght4(h)
        h = self.output(h)
        s = F.sigmoid(h)
        return s


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


if __name__ == '__main__':
    root_path = "/opt/vqa-data"
    dataset = VqaDataset(root_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
    embedding_model_path = "/opt/vqa-data/wiki.en.genism"
    fast_text = KeyedVectors.load(embedding_model_path)
    # initial_weights = torch.from_numpy(fast_text[fast_text.wv.vocab])
    initial_weights = torch.randn(dataset.questions_vocab_size, 300).cuda()
    temp = dataset.questions_vocab()
    for i in range(len(temp)):
        try:
            initial_weights[i, :] = torch.from_numpy(fast_text[temp[i]]).cuda()
        except:
            initial_weights[i, :] = torch.randn(1, 300).cuda()
    print("finish weight init")
    net = Network(dataset.questions_vocab_size, dataset.answers_vocab_size, initial_weights)
    net = nn.DataParallel(net).cuda()
    criterion = MultiLabelSoftMarginLoss().cuda()
    optimizer = Adam(net.parameters(), lr=0.001)
    epochs = 20
    dataset.dispose()
    del fast_text
    for epoch in range(epochs):
        losses = []
        for batch, (questions, image_features, answers) in enumerate(dataloader, 0):
            questions, image_features, answers = Variable(questions.cuda()), Variable(
                image_features.cuda()), Variable(answers.cuda())
            optimizer.zero_grad()
            outputs = net(questions, image_features)
            loss = criterion(outputs, answers)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean())
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        })
        print('[{}/{}] Loss: {}'.format(epoch + 1, epochs, np.mean(losses)))
