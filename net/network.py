import os
from multiprocessing import cpu_count

import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from torch.autograd import Variable
from torch.cuda import manual_seed
from torch.nn import GRU, Linear, Embedding, DataParallel, Dropout, BatchNorm1d, BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adadelta
from torch.utils.data import DataLoader

from dataset.vqa_dataset import VqaDataset, DataType
from embedder.image_embedder import ImageEmbedder
from embedder.sentence_embedder import SentenceEmbedder
from net.gated_hyperbolic_tangent import GatedHyperbolicTangent
from util.preprocess import to_module, save_checkpoint, print_size


def create_batch_dir(batch_size: int, base_dir: str = "../models"):
    path = "{}/batch-{}-models".format(base_dir, batch_size)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load(model_path: str, info, use_cuda=True):
    state = torch.load(model_path)
    state = state["state_dict"]
    state = to_module(state)
    net = Network(info["question_vocab_size"] + 1, info["answer_vocab_size"] + 1)
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
    embeds = torch.randn(l + 2, 300)
    embeds[0, :] = torch.zeros(1, 300)
    for i in range(2, l + 2):
        try:
            embeds[i, :] = torch.from_numpy(pretrained_embed_model[vocab[i]]).view(1, 300)
        except:
            embeds[i, :] = torch.zeros(1, 300)
    return embeds


def load_words_images(root_path: str, vocabs) -> torch.FloatTensor:
    from util.vocab_images_features_extractor import VocabImagesFeaturesExtractor
    extractor = VocabImagesFeaturesExtractor()
    return extractor.load(root_path, vocabs)


class Network(nn.Module):
    def __init__(self, question_vocab_size: int, answer_vocab_size: int, question_max_length: int = 14,
                 initial_embedding_weights=None,
                 word_vector_length: int = 300,
                 image_location_number: int = 36, image_features_size: int = 2048, embedding_size: int = 512,
                 initial_output_embed_weights=None, initial_output_images_weights=None, reg: bool = True):
        super().__init__()
        self.image_location_number = image_location_number
        self.question_max_length = question_max_length
        self.embedding_size = embedding_size
        self.word_vector_length = word_vector_length
        self.embedding = Embedding(question_vocab_size, word_vector_length)
        self.reg = reg
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
        if reg:
            # some reg layers
            self.batch_norm1, self.batch_norm2 = BatchNorm1d(image_location_number), BatchNorm1d(
                image_features_size)
            self.dropout = Dropout(p=0.5)

        # output
        self.output_modified = initial_output_embed_weights is not None
        if self.output_modified:
            self.output_image_ght = GatedHyperbolicTangent(embedding_size, 2048)
            self.output_image = Linear(2048, answer_vocab_size)
            self.output_image.weight.copy_(initial_output_images_weights)
            self.output_text_ght = GatedHyperbolicTangent(embedding_size, 300)
            self.output_text = Linear(300, answer_vocab_size)
            self.output_text.weight.copy_(initial_output_embed_weights)
        else:
            self.output = Linear(embedding_size, answer_vocab_size)

    def forward(self, question, image):
        # sentence embedding
        hidden = self.question_embedding(question)
        q = self.question_ght(hidden)
        # image embedding
        image = F.normalize(image)
        # fused features
        fusion_features = self.fusion(hidden, image)
        # attentions
        v = self.attentions(fusion_features, image)
        # can be replaced with *
        # joint
        h = torch.mul(q, v)  # size = (batch size, embedding size)
        if self.reg:
            h = self.dropout(h)
        # output
        return self.out(h)

    def question_embedding(self, question):
        print_size("question", question)
        question_vectors = self.embedding(
            question)  # size = (batch size, number of words for each sentence, word vector length = 300)
        print_size("question vectors", question_vectors)
        question_vectors = self.pad_input(
            question_vectors)  # size = (batch size, max question length, word vector length)
        out, hidden = self.gru(question_vectors.permute(1, 0, 2))
        hidden = hidden[-1]  # size = (batch size, embedding size = 512)
        # hidden = out[-1]
        return hidden

    def fusion(self, hidden, image):
        # concat
        hidden_mat = hidden.repeat(1,
                                   self.image_location_number)  # size = (batch size, image location number * embedding size)
        hidden_mat = hidden_mat.view(-1, self.image_location_number,
                                     self.embedding_size)  # size = (batch size, image location number, embedding size)
        fusion_features = torch.cat((image, hidden_mat),
                                    -1)  # size = (batch size, image location number, embedding size + image features size)
        # print_size("fusion features1", fusion_features)
        fusion_features = self.fusion_ght(fusion_features)  # size = (batch size, image location number, embedding size)
        if self.reg:
            fusion_features = self.batch_norm1(fusion_features)
        return fusion_features

    def attentions(self, fusion_features, image):
        attentions = self.attention_layer(fusion_features)
        attentions = F.softmax(attentions.squeeze())  # size = (batch size, image location number)
        temp = attentions.unsqueeze(1)
        v = torch.bmm(temp, image).squeeze()  # size = (batch size, image features size)
        if self.reg:
            v = self.batch_norm2(v)
        v = self.v_ght(v)
        return v

    def out(self, h):
        if not self.output_modified:
            h = self.output_ght(h)
            h = self.output(h)
        else:
            image = self.output_image_ght(h)
            image = self.output_image(image)
            text = self.output_text_ght(h)
            text = self.output_text(text)
            h = image + text
        return h

    def pad_input(self, x):
        for i in range(x.size(0)):
            question = x[i, :, :]
            padding_size = self.question_max_length - question.size(0)
            if padding_size > 0:
                question = torch.cat((question, torch.zeros(padding_size, self.word_vector_length)))
            elif padding_size < 0:
                question = question[:self.question_max_length, :]
            x[i, :, :] = question
        return x


if __name__ == '__main__':
    manual_seed(1000)
    root_path = "/opt/vqa-data"
    soft_max, modified_output, glove, reg = True, False, True, False
    batch_size = 512
    base_dir = create_batch_dir(batch_size)
    dataset = VqaDataset(root_path, soft_max, type=DataType.ALL)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    embedding_model_path = "/opt/vqa-data/wiki.en.genism" if not glove else "/opt/vqa-data/gensim_glove_vectors.txt"
    embedding_model = KeyedVectors.load(embedding_model_path) if not glove else KeyedVectors.load_word2vec_format(
        embedding_model_path)
    initial_embed_weights = load_words_embed(embedding_model, dataset.questions_vocab())
    initial_output_weights, initial_output_images_weights = None, None
    print("finish weight init")
    if modified_output:
        t = dataset.answers_vocab()
        initial_output_weights = load_words_embed(embedding_model, t)
        initial_output_images_weights = load_words_images(root_path, t)
    net = Network(dataset.questions_vocab_size + 2, dataset.answers_vocab_size + 1, initial_embed_weights,
                  initial_output_embed_weights=initial_output_weights,
                  initial_output_images_weights=initial_output_images_weights, reg=reg)
    net = DataParallel(net).cuda()
    criterion = (BCEWithLogitsLoss() if not soft_max else CrossEntropyLoss()).cuda()
    # criterion = VqaLoss().cuda()
    optimizer = Adadelta(net.parameters(), lr=0.001)
    epochs = 20
    print("begin training")
    batches = dataset.number_of_questions() / batch_size
    for epoch in range(epochs):
        epoch_loss, epoch_correct = 0, 0
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
            if batch % 40 == 0:
                print('Epoch %02d(%03d/%03d), loss: %.3f, correct: %3d / %d (%.2f%%)' %
                      (epoch + 1, batch, batches, loss.data[0], correct, batch_size, correct * 100 / batch_size))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch, base_dir)
        print('Epoch {} done, average loss: {}, average accuracy: {}%'.format(
            epoch + 1, epoch_loss / batches, epoch_correct * 100 / (batches * batch_size)))
