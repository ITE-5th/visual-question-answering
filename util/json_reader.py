import json


class JsonReader:
    def __init__(self, file_path: str):
        with open(file_path) as f:
            self.data = json.load(f)

    def get(self, key):
        return self.data[key]

    def __getitem__(self, item):
        return self.get(item)
