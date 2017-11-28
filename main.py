import torch
from torch.nn import DataParallel

from dataset.vqa_dataset import VqaDataset
from net.network import Network
from embedder.image_embedder import ImageEmbedder
from embedder.sentence_embedder import SentenceEmbedder

def to_module(state_dict):
    new_state_dict = dict()
    for key in state_dict.keys():
        new_name = key[key.index(".") + 1:]
        new_state_dict[new_name] = state_dict[key]
    return new_state_dict


def start():
    image_embedder = ImageEmbedder("")
    sentence_embedder = SentenceEmbedder("")
    root_dir = "/opt/vqa-data"
    info = VqaDataset.load_eval_info(root_dir)
    image_path = "image.png"
    image = image_embedder.embed(image_path)
    image = image.unsequeeze(0)
    question = input("Enter your question:")
    question = sentence_embedder.embed(question)
    question = question.unsequeeze(0)
    model_path = "models/batch-512-models/epoch-100-checkpoint.pth.tar"
    state = torch.load(model_path)
    state = state["state_dict"]
    state = to_module(state)
    net = Network(info["question_vocab_size"], info["answer_vocab_size"])
    net.load_state_dict(state)
    net = DataParallel(net).cuda()
    net.eval()
    result = net(question, image)
    result = result.sequeeze(0).view(info["answer_vocab_size"])
    _, predicted = result.data.topk(10)
    predicted = predicted.view(10)
    answers = info["answer_vocab"]
    for pred in predicted:
        print(answers[pred])


if __name__ == '__main__':
    start()
