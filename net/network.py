import os
from multiprocessing import cpu_count

import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from torch.autograd import Variable
from torch.nn import GRU, Linear, Embedding, DataParallel, BCEWithLogitsLoss, CrossEntropyLoss, Dropout
from torch.optim import Adadelta
from torch.utils.data import DataLoader

from dataset.vqa_dataset import VqaDataset
from embedder.image_embedder import ImageEmbedder
from embedder.sentence_embedder import SentenceEmbedder
from net.gated_hyperbolic_tangent import GatedHyperbolicTangent
from util.preprocess import to_module, save_checkpoint
from util.vocab_images_features_extractor import VocabImagesFeaturesExtractor


def create_batch_dir(batch_size: int, base_dir: str = "../models"):
    path = "{}/batch-{}-models".format(base_dir, batch_size)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load(model_path: str, soft_max: bool, info, use_cuda=True):
    state = torch.load(model_path)
    state = state["state_dict"]
    state = to_module(state)
    net = Network(soft_max, info["question_vocab_size"] + 1, info["answer_vocab_size"] + 1)
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
    return [answers[pred] for pred in predicted if pred != 0]


def load_words_embed(pretrained_embed_model, vocab) -> torch.FloatTensor:
    l = len(vocab)
    embeds = torch.randn(l + 1, 300)
    for i in range(l):
        try:
            embeds[i + 1, :] = torch.from_numpy(pretrained_embed_model[vocab[i]]).view(1, 300)
        except:
            embeds[i + 1, :] = torch.randn(1, 300)
    return embeds


def load_words_images(root_path: str, vocabs) -> torch.FloatTensor:
    extractor = VocabImagesFeaturesExtractor()
    return extractor.load(root_path, vocabs)


class Network(nn.Module):
    def __init__(self, soft_max: bool, question_vocab_size: int, answer_vocab_size: int, initial_embedding_weights=None,
                 word_vector_length: int = 300,
                 image_location_number: int = 36, image_features_size: int = 2048, embedding_size: int = 512,
                 initial_output_embed_weights=None, initial_output_images_weights=None):
        super().__init__()
        self.soft_max = soft_max
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
        self.dropout = Dropout()
        self.output_modified = initial_output_embed_weights is not None
        if self.output_modified:
            self.output_image_ght = GatedHyperbolicTangent(2048, embedding_size)
            self.output_image = Linear(embedding_size, answer_vocab_size)
            self.output_text_ght = GatedHyperbolicTangent(300, embedding_size)
            self.output_text = Linear(embedding_size, answer_vocab_size)
        else:
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
        image = F.normalize(image, -1)
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
        result = self.dropout(h)
        if not self.output_modified:
            result = self.output_ght(result)
            result = self.output(result)
        else:
            im = self.output_image_ght(result)
            im = self.output_image(im)
            text = self.output_text_ght(result)
            text = self.output_text(text)
            result = im + text
        # print_size("result", result)
        return result


if __name__ == '__main__':
    root_path = "/opt/vqa-data"
    soft_max = True
    modified_output = False
    batch_size = 512
    base_dir = create_batch_dir(batch_size)
    dataset = VqaDataset(root_path, soft_max)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    print("questions number = " + str(len(dataset)))
    embedding_model_path = "/opt/vqa-data/wiki.en.genism"
    fast_text = KeyedVectors.load(embedding_model_path)
    initial_embed_weights = load_words_embed(fast_text, dataset.questions_vocab())
    initial_output_weights = None
    initial_output_images_weights = None
    print("finish weight init")
    if modified_output:
        initial_output_weights = load_words_embed(fast_text, dataset.answers_vocab())
        initial_output_images_weights = load_words_images(root_path, dataset.answers_vocab())
    net = Network(soft_max, dataset.questions_vocab_size + 1, dataset.answers_vocab_size + 1, initial_embed_weights,
                  initial_output_embed_weights=initial_output_weights,
                  initial_output_images_weights=initial_output_images_weights)
    net = DataParallel(net).cuda()
    criterion = (BCEWithLogitsLoss() if not soft_max else CrossEntropyLoss()).cuda()
    optimizer = Adadelta(net.parameters(), lr=0.01)
    epochs = 100
    print("begin training")
    batches = dataset.number_of_questions() / batch_size
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_correct = 0
        for batch, (questions, image_features, answers) in enumerate(dataloader, 0):
            questions, image_features, answers = Variable(questions.cuda()), Variable(
                image_features.cuda()), Variable(answers.cuda())
            outputs = net(questions, image_features)
            loss = criterion(outputs, answers)
            if not soft_max:
                _, correct_indices = torch.max(answers.data, 1)
                _, expected_indices = torch.max(outputs.data, 1)
                correct = torch.eq(correct_indices, expected_indices).sum()
            else:
                _, first = outputs.data.max(1)
                second = answers.data
                correct = torch.eq(first, second).sum()
            epoch_correct += correct
            epoch_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("finish batch number = {}".format(batch + 1))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch, base_dir)
        print('Epoch {} done, average loss: {}, average accuracy: {}%'.format(
            epoch + 1, epoch_loss / batches, epoch_correct * 100 / (batches * batch_size)))
