import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class VqaDataset(Dataset):
    def __init__(self, root_path: str, question_length: int = 14):
        self.question_length = question_length
        with open("{}/train_q_dict.p".format(root_path), "rb") as f:
            temp = pickle.load(f)
            self.questions_indices_to_words = temp["itow"]
            self.questions_words_to_indices = temp["wtoi"]
            self.questions_vocab_size = len(temp["itow"])
        with open("{}/train_a_dict.p".format(root_path), "rb") as f:
            temp = pickle.load(f)
            self.answers_indices_to_words = temp["itow"]
            self.answers_words_to_indices = temp["wtoi"]
            self.answers_vocab_size = len(temp["itow"])
        print("finish extraction of word to indices")
        with open("{}/vqa_train_final.json".format(root_path), "r") as f:
            self.questions = json.load(f)
            # with Pool(cpu_count()) as p:
            #     self.questions = p.map(self.extract_question_features, self.questions)
            #     p.close()
            #     p.join()
            self.questions = [self.extract_question_features(question) for question in self.questions]
        print("finish extraction of questions")
        self.images_path = root_path + "/images/"
        image_ids = [path[:path.rfind(".")] for path in os.listdir(self.images_path)]
        # with Pool(cpu_count()) as p:
        #     self.images_features = dict(p.map(self.extract_image_features, image_ids))
        #     p.close()
        #     p.join()
        self.images_features = {int(image_id): self.extract_image_features(image_id) for image_id in image_ids}
        print("finish extraction images")

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        image_feature = self.images_features[question[1]]
        return question[0], image_feature, question[2]

    def extract_question_features(self, question):
        # print(question)
        question_indices = [0] * self.question_length
        for i, word in enumerate(question["question_toked"]):
            try:
                question_indices[i] = self.questions_words_to_indices[word]
            except:
                question_indices[i] = 0
        answers = [0] * self.answers_vocab_size
        for word, score in question["answers_w_scores"]:
            ind = self.answers_words_to_indices[word]
            answers[ind] = float(score)
        return torch.LongTensor(question_indices), int(question["image_id"]), torch.FloatTensor(answers)

    def extract_image_features(self, image_id: str):
        arr = np.load(self.images_path + image_id + ".npy")
        if arr.shape != (36, 2048):
            print("what the fuck")
        return torch.from_numpy(arr)

    def questions_vocab(self):
        return list(self.questions_words_to_indices.keys())

    def dispose(self):
        del self.images_path, self.answers_words_to_indices, self.answers_indices_to_words, self.questions_indices_to_words, self.questions_words_to_indices, self.questions_vocab_size
