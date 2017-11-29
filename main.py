from dataset.vqa_dataset import VqaDataset
from net.network import load, predict


def start():
    root_dir = "/opt/vqa-data"
    info = VqaDataset.load_eval_info(root_dir)
    image_path = "image.png"
    question = input("Enter your question:")
    model_path = "models/batch-512-models/epoch-100-checkpoint.pth.tar"
    net = load(model_path, info)
    predicted = predict(net, info, question, image_path)
    print(predicted)


if __name__ == '__main__':
    start()
