from embedder.embedder import Embedder


class ImageEmbedder(Embedder):
    def __init__(self, model_path: str):
        self.embedder = model_path

    def embed(self, x):
        return self.embedder.forward(x)
