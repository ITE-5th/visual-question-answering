import os
import json
import pickle
import string
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from multiprocessing import Pool, cpu_count


class VqaDataset(Dataset):
    def __init__(self, root_path: str):
        self.images_path = root_path + "/images/"
        image_ids = [path[:path.rfind(".")] for path in os.listdir(self.images_path)]
        with Pool(cpu_count()) as p:
            self.images_features = dict(p.map(self.extract_image_features, image_ids))
        with open("../data/vqa_train_final.json", "r") as f:
            self.questions = json.load(f)
        with open("../data/train_q_dict.p", "rb") as f:
            temp = pickle.load(f)
            self.questions_indices_to_words = temp["itow"]
            self.questions_words_to_indices = temp["wtoi"]
            self.questions_vocab_size = len(temp["itow"])
        with open("../data/train_a_dict.p", "rb") as f:
            temp = pickle.load(f)
            self.answers_indices_to_words = temp["itow"]
            self.answers_words_to_indices = temp["wtoi"]
            self.answers_vocab_size = len(temp["itow"])

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        question_indices = [self.answers_words_to_indices[word] for word in question["question_toked"] if word not in string.punctuation]
        # TODO : wrong output
        answers = [0] * len(self.answers_words_to_indices)
        for answer in question["answers"]:
            ind = self.answers_words_to_indices[answer[0]]
            answers[ind] = float(answer[1]) / 10
        image_feature = self.images_features[int(question["image_id"])]
        return torch.LongTensor(question_indices), image_feature, torch.FloatTensor(answers)

    def extract_image_features(self, image_id: str):
        arr = np.load(self.images_path + image_id + ".npy")
        return int(image_id), torch.from_numpy(arr)

    def questions_vocab(self):
        return self.questions_words_to_indices.keys()