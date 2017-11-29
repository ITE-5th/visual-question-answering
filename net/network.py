import os
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from torch.autograd import Variable
from torch.nn import MultiLabelSoftMarginLoss, GRU, Linear, Embedding, DataParallel
from torch.optim import Adadelta
from torch.utils.data import DataLoader

from dataset.vqa_dataset import VqaDataset
from embedder.image_embedder import ImageEmbedder
from embedder.sentence_embedder import SentenceEmbedder
from net.gated_hyperbolic_tangent import GatedHyperbolicTangent


def to_module(state_dict):
    new_state_dict = dict()
    for key in state_dict.keys():
        new_name = key[key.index(".") + 1:]
        new_state_dict[new_name] = state_dict[key]
    return new_state_dict


def print_size(name, tensor):
    print(name + " = " + str(tensor.size()))


def create_batch_dir(batch_size: int, base_dir: str = "../models"):
    path = "{}/batch-{}-models".format(base_dir, batch_size)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_checkpoint(state, epoch: int, directory: str = '../models'):
    torch.save(state, "{}/epoch-{}-checkpoint.pth.tar".format(directory, epoch + 1))


def load(model_path: str, info, use_cuda=True):
    state = torch.load(model_path)
    state = state["state_dict"]
    state = to_module(state)
    net = Network(info["question_vocab_size"] + 1, info["answer_vocab_size"])
    net.load_state_dict(state)
    net = DataParallel(net)
    if use_cuda:
        net = net.cuda()
    net.eval()
    return net


def predict(net, info, question, image_path):
    image_embedder = ImageEmbedder("")
    sentence_embedder = SentenceEmbedder(info["question_vocab"])
    image = image_embedder.embed(image_path)
    image = image.unsequeeze(0)
    question = sentence_embedder.embed(question)
    question = question.unsequeeze(0)
    result = net(question, image)
    result = result.sequeeze(0).view(info["answer_vocab_size"])
    _, predicted = result.data.topk(10)
    predicted = predicted.view(10)
    answers = info["answer_vocab"]
    return [answers[pred] for pred in predicted]


class Network(nn.Module):
    def __init__(self, question_vocab_size: int, answer_vocab_size: int, initial_embedding_weights=None,
                 word_vector_length: int = 300,
                 image_location_number: int = 36, image_features_size: int = 2048, embedding_size: int = 512):
        super().__init__()
        self.image_location_number = image_location_number
        self.embedding_size = embedding_size
        self.embedding = Embedding(question_vocab_size, word_vector_length)
        if initial_embedding_weights is not None:
            self.embedding.weight.data.copy_(initial_embedding_weights)
        self.gru = GRU(word_vector_length, embedding_size)
        # fusion feature ght
        self.fusion_ght = GatedHyperbolicTangent(embedding_size + image_features_size, embedding_size)
        self.attention_layer = Linear(embedding_size, 1)
        # question ght
        self.question_ght = GatedHyperbolicTangent(embedding_size, embedding_size)
        # attention result ght
        self.v_ght = GatedHyperbolicTangent(image_features_size, embedding_size)
        # output ght
        self.output_ght = GatedHyperbolicTangent(embedding_size, embedding_size)
        # output
        self.output = Linear(embedding_size, answer_vocab_size)

    def forward(self, question, image):
        # sentence embedding
        question_vectors = self.embedding(
            question)  # size = (batch size, max sentence length, word_vector_length = 300)
        out, hidden = self.gru(question_vectors.permute(1, 0, 2))
        hidden = hidden[-1]  # size = (batch size, embedding size = 512)
        # print_size("hidden", hidden)
        # image embedding
        # dim = -1 ?
        image = F.normalize(image)
        # concat
        hidden_mat = hidden.repeat(1,
                                   self.image_location_number)  # size = (batch size, image location number * embedding size)
        hidden_mat = hidden_mat.view(-1, self.image_location_number,
                                     self.embedding_size)  # size = (batch size, image location number, embedding size)
        fusion_features = torch.cat((image, hidden_mat),
                                    -1)  # size = (batch size, image location number, embedding size + image features size)
        # print_size("fusion features1", fusion_features)
        fusion_features = self.fusion_ght(fusion_features)  # size = (batch size, image location number, embedding size)
        # print_size("fusion features2", fusion_features)
        # attentions
        attentions = self.attention_layer(fusion_features)
        attentions = F.softmax(attentions.squeeze())  # size = (batch size, image location number)
        temp = attentions.unsqueeze(1)
        v = torch.bmm(temp, image).squeeze()  # size = (batch size, image features size)
        # print_size("v", v)
        # joint
        v = self.v_ght(v)
        q = self.question_ght(hidden)
        # can be replaced with *
        h = torch.mul(q, v)  # size = (batch size, embedding size)
        # print_size("h", h)
        # output
        result = self.output_ght(h)
        result = self.output(result)
        result = F.sigmoid(result)
        # print_size("result", result)
        return result


if __name__ == '__main__':
    root_path = "/opt/vqa-data"
    batch_size = 512
    base_dir = create_batch_dir(batch_size)
    dataset = VqaDataset(root_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    print("questions number = " + str(len(dataset)))
    embedding_model_path = "/opt/vqa-data/wiki.en.genism"
    fast_text = KeyedVectors.load(embedding_model_path)
    # initial_weights = torch.from_numpy(fast_text[fast_text.wv.vocab])
    initial_weights = torch.randn(dataset.questions_vocab_size + 1, 300)
    temp = dataset.questions_vocab()
    for i in range(len(temp)):
        try:
            initial_weights[i + 1, :] = torch.from_numpy(fast_text[temp[i]]).view(1, 300)
        except:
            initial_weights[i + 1, :] = torch.randn(1, 300)
    print("finish weight init")
    net = Network(dataset.questions_vocab_size + 1, dataset.answers_vocab_size, initial_weights)
    net = DataParallel(net).cuda()
    criterion = MultiLabelSoftMarginLoss().cuda()
    optimizer = Adadelta(net.parameters())
    epochs = 100
    print("begin training")
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
            print("finish batch number = {}".format(batch + 1))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch, base_dir)
        print('[{}/{}] Loss: {}'.format(epoch + 1, epochs, np.mean(losses)))
