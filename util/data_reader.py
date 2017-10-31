from itertools import chain

import torch

from embedder.image_embedder import ImageEmbedder
from embedder.sentence_embedder import SentenceEmbedder
from util.json_reader import JsonReader


class DataReader:
    def __init__(self, sentence_embedder: SentenceEmbedder, image_embedder: ImageEmbedder):
        self.sentence_embedder, self.image_embedder = sentence_embedder, image_embedder

    def read(self, data_dir_path: str) -> torch.FloatTensor:
        train_questions = JsonReader("{}/{}".format(data_dir_path, "train_questions.json")).get("questions")
        val_questions = JsonReader("{}/{}".format(data_dir_path, "val_questions.json")).get("questions")
        train_annotations = JsonReader("{}/{}".format(data_dir_path, "train_annotations.json")).get("annotations")
        val_annotations = JsonReader("{}/{}".format(data_dir_path, "val_annotations.json")).get("annotations")
        questions = {}
        for question in chain(train_questions, val_questions):
            question_id = question.question_id
            if question_id not in questions:
                questions[question_id] = []
            questions[question_id].append((question.image_id, question.question))
        for annotation in chain(train_annotations, val_annotations):
            pass
