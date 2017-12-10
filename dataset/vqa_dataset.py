import json
import os
import pickle
from enum import Enum
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class DataType(Enum):
    TRAIN = 0,
    TEST = 1,
    ALL = 2


class VqaDataset(Dataset):
    def __init__(self, root_path: str, soft_max: bool = False, type: DataType = DataType.TRAIN,
                 question_length: int = 14):
        self.soft_max = soft_max
        self.type = type
        self.question_length = question_length
        with open("{}/train_q_dict.p".format(root_path), "rb") as f:
            temp = pickle.load(f)
            self.questions_indices_to_words = temp["itow"]
            self.questions_words_to_indices = temp["wtoi"]
            self.questions_vocab_size = len(self.questions_indices_to_words)
        with open("{}/train_a_dict.p".format(root_path), "rb") as f:
            temp = pickle.load(f)
            self.answers_indices_to_words = temp["itow"]
            self.answers_words_to_indices = temp["wtoi"]
            self.answers_vocab_size = len(self.answers_indices_to_words)
        print("finish word to indices extraction")
        if self.type == DataType.TRAIN or self.type == DataType.TRAIN:
            with open("{}/vqa_{}_final.json".format(root_path, "train" if type == DataType.TRAIN else "val"), "r") as f:
                self.questions = json.load(f)
                self.questions = [self.extract_question_features(question) for question in self.questions]
        else:
            with open("{}/vqa_train_final.json".format(root_path), "r") as f:
                temp = json.load(f)
                self.questions = [self.extract_question_features(question) for question in temp]
            with open("{}/vqa_val_final.json".format(root_path), "r") as f:
                temp = json.load(f)
                self.questions += [self.extract_question_features(question) for question in temp]
        self.length = len(self.questions)
        with Pool(cpu_count()) as p:
            self.features = p.map(self.extract_features, list(range(self.length)))
            p.close()
            p.join()
        print("finish questions extraction")
        print("questions = {}".format(len(self.questions)))
        self.images_path = root_path + "/images/"
        image_ids = [path[:path.rfind(".")] for path in os.listdir(self.images_path)]
        self.images_features = {int(image_id): self.extract_image_features(image_id) for image_id in image_ids}
        print("finish images extraction")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.features[index]

    def extract_features(self, index):
        question = self.questions[index]
        image_feature = self.images_features[question[1]]
        return question[0], image_feature, question[2]

    def extract_question_features(self, question):
        question_indices = [0] * self.question_length
        for i, word in enumerate(question["question_toked"]):
            try:
                question_indices[i] = self.questions_words_to_indices[word]
            except:
                question_indices[i] = 0
        if not self.soft_max:
            answers = [0] * (self.answers_vocab_size + 1)
            temp = question["answers_w_scores"]
            for word, score in temp:
                ind = self.answers_words_to_indices[word] + 1
                answers[ind] = float(score)
            answers = torch.FloatTensor(answers)
        else:
            try:
                answers = self.answers_words_to_indices[question["answer"]] + 1
            except:
                answers = 0
        return torch.LongTensor(question_indices), int(question["image_id"]), answers

    def extract_image_features(self, image_id: str):
        arr = np.load(self.images_path + image_id + ".npy")
        return torch.from_numpy(arr)

    def number_of_questions(self):
        return len(self.questions)

    def questions_vocab(self):
        return list(self.questions_words_to_indices.keys())

    def answers_vocab(self):
        return list(self.answers_words_to_indices.keys())

    def dispose(self):
        del self.images_path, self.answers_words_to_indices, self.answers_indices_to_words, self.questions_indices_to_words, self.questions_words_to_indices, self.questions_vocab_size

    @staticmethod
    def load_eval_info(root_dir: str):
        res = {}
        with open("{}/train_q_dict.p".format(root_dir), "rb") as f:
            temp = pickle.load(f)
            t = temp["wtoi"].keys()
            res["question_vocab"] = list(t)
            res["question_vocab_size"] = len(t)
        with open("{}/train_a_dict.p".format(root_dir), "rb") as f:
            temp = pickle.load(f)
            t = temp["wtoi"].keys()
            res["answer_vocab"] = list(t)
            res["answer_vocab_size"] = len(t)
        return res

    @staticmethod
    def load_answers_vocab(root_dir: str):
        with open("{}/train_a_dict.p".format(root_dir), "rb") as f:
            temp = pickle.load(f)
            return list(temp["wtoi"].keys())

    def load_info(self):
        info = {"question_vocab_size": self.questions_vocab_size,
                "question_vocab": list(self.questions_words_to_indices.keys()),
                "answer_vocab_size": self.answers_vocab_size,
                "answer_vocab": list(self.answers_words_to_indices.keys())}
        return info
