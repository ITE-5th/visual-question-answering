from multiprocessing import cpu_count

import torch
from torch.autograd import Variable
from torch.nn import MultiLabelSoftMarginLoss
from torch.utils.data import DataLoader

from dataset.vqa_dataset import VqaDataset
from net.network import load

if __name__ == '__main__':
    root_path = "/opt/vqa-data"
    model_path = "../models/batch-512-models/epoch-100-checkpoint.pth.tar"
    batch_size = 2048
    dataset = VqaDataset(root_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    info = dataset.load_info()
    net = load(model_path, info)
    criterion = MultiLabelSoftMarginLoss().cuda()
    total_error = 0
    accuracy = 0
    for batch, (questions, image_features, answers) in enumerate(dataloader, 0):
        questions, image_features, answers = Variable(questions.cuda()), Variable(
            image_features.cuda()), Variable(answers.cuda())
        outputs = net(questions, image_features)
        _, correct_indices = torch.max(answers.data, 1)
        _, expected_indices = torch.max(outputs.data, 1)
        net.zero_grad()
        loss = criterion(outputs, answers)
        total_error += loss.data.sum()
        accuracy += (correct_indices == expected_indices).sum()
    accuracy /= dataset.number_of_questions()
    accuracy *= 100
    print("error = {}".format(total_error))
    print("accuracy = {}%".format(accuracy))
