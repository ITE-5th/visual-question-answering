from multiprocessing import cpu_count

import torch
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader

from dataset.vqa_dataset import VqaDataset
from net.network import load

if __name__ == '__main__':
    soft_max = True
    root_path = "/opt/vqa-data"
    model_path = "../models/batch-512-models/epoch-100-checkpoint.pth.tar"
    batch_size = 2048
    dataset = VqaDataset(root_path, soft_max)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    info = dataset.load_info()
    net = load(model_path, soft_max, info)
    criterion = (CrossEntropyLoss() if not soft_max else BCEWithLogitsLoss()).cuda()
    epoch_correct, epoch_loss = 0, 0
    batches = dataset.number_of_questions() / batch_size
    for batch, (questions, image_features, answers) in enumerate(dataloader, 0):
        questions, image_features, answers = Variable(questions.cuda()), Variable(
            image_features.cuda()), Variable(answers.cuda())
        outputs = net(questions, image_features)
        loss = criterion(outputs, answers)
        if not soft_max:
            _, correct_indices = torch.max(answers.data, 1)
            _, expected_indices = torch.max(outputs.data, 1)
            correct = torch.eq(correct_indices, expected_indices).sum()
        else:
            _, first = outputs.data.max(1)
            second = answers.data
            correct = torch.eq(first, second).sum()
        epoch_correct += correct
        epoch_loss += loss.data[0]
        net.zero_grad()
    print("loss = {}, accuracy = {}".format(epoch_loss / batches, epoch_correct * 100 / (batches * batch_size)))
