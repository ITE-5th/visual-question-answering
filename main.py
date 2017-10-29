from embedder.sentence_embedder import SentenceEmbedder


def start():
    sentence = "small test"
    embedder = SentenceEmbedder("data/GoogleNews-vectors-negative300.bin")
    print(embedder.embed(sentence))


if __name__ == '__main__':
    start()