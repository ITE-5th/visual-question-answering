import os

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Sequential

from dataset.vqa_dataset import VqaDataset

model = torch.load("../data/resnet101.pth").cuda()

for param in model.parameters():
    param.requires_grad = False
model = Sequential(*list(model.modules())[:-1])


class VocabImagesFeaturesExtractor:

    def extract(self, root_dir: str, vocabs):
        path = "{}/output_images_features".format(root_dir)
        if not os.path.exists(path):
            os.makedirs(path)
        for i in range(len(vocabs)):
            vocab = vocabs[i]
            temp = torch.zero(1, 2048)
            vocab_images_path = "{}/output_vocab_images".format(root_dir)
            vocab_features = "{}/{}".format(path, vocab)
            vocab_images = ["{}/{}".format(vocab_images_path, p) for p in os.listdir(vocab_images_path)]
            for j in range(len(vocab_images)):
                vocab_image = vocab_images[j]
                image = cv2.imread(vocab_image)
                image = cv2.resize(image, (224, 244))
                image = np.swapaxes(image, 0, 2)
                image = np.swapaxes(image, 1, 2)
                # resnet101 input
                image = Variable(torch.from_numpy(image).cuda().unsqueeze(0))
                temp += model(image).squeeze(0).view(1, 2048)
            temp /= 10
            torch.save(temp, vocab_features)

    def load(self, root_dir: str, vocabs):
        path = "{}/output_images_features".format(root_dir)
        temp = os.listdir(path)
        features = {p: torch.load("{}/{}".format(path, p)) for p in temp}
        result = torch.FloatTensor(len(vocabs) + 1, 2048)
        for i in range(len(vocabs)):
            vocab = vocabs[i]
            feature = features[vocab]
            result[i + 1, :] = feature
        return result


if __name__ == '__main__':
    root_dir = "../data"
    ext = VocabImagesFeaturesExtractor()
    vocab = VqaDataset.load_answers_vocab(root_dir)
    ext.extract(root_dir, vocab)
