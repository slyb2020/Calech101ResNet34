import torch
from torch import nn
from torchvision.models import alexnet
from Caltech101Dataset import caltech101Dataset, trainLoader
import tqdm
import time


# alexNet = alexnet()
# print(alexNet)
class Caltech101AlexNet(nn.Module):
    def __init__(self, preTrain=False):
        super(Caltech101AlexNet, self).__init__()
        if preTrain:
            self.alexNet = alexnet(pretrained=True)
        else:
            self.alexNet = alexnet()
        for layer in self.alexNet.children():
            print(layer)
            layer.requires_grad = False
        self.alexNet.add_module("add_linear", nn.Linear(1000, 101))
        # self.addLinearLayer = nn.Linear(1000,101)

    def forward(self, x):
        x = self.alexNet(x)
        # x = self.addLinearLayer(x)
        return x

class Caltech101AlexNet(nn.Module):
    def __init__(self, preTrain=False):
        super(Caltech101AlexNet, self).__init__()
        if preTrain:
            self.alexNet = alexnet(pretrained=True)
        else:
            self.alexNet = alexnet()
        for layer in self.alexNet.children():
            print(layer)
            layer.requires_grad = False
        self.alexNet.add_module("add_linear", nn.Linear(1000, 101))
        # self.addLinearLayer = nn.Linear(1000,101)

    def forward(self, x):
        x = self.alexNet(x)
        # x = self.addLinearLayer(x)
        return x


if __name__ == "__main__":
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    dataAmount = caltech101Dataset.__len__()
    maxEpoch = 10
    caltech101AlexNet = Caltech101AlexNet(preTrain=True)
    caltech101AlexNet = torch.load("./model/AlexNet.pkl")
    caltech101AlexNet.to(device)
    # print(caltech101AlexNet)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(caltech101AlexNet.parameters(), lr=1e-3, momentum=0.9)
    steps = 0
    startTime = time.time()
    for epoch in range(maxEpoch):
        caltech101AlexNet.train()
        torch.enable_grad()
        lossEpoch = 0
        for images, labels in trainLoader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predictions = caltech101AlexNet(images)
            loss = criterion(predictions, labels)
            lossEpoch += loss
            loss.backward()
            optimizer.step()
        endTime = time.time()
        print("经{}代学习, 当前损失值为{},用时:{}".format(epoch + 25, lossEpoch, endTime - startTime))
        torch.save(caltech101AlexNet, "./model/AlexNet.pth")
        caltech101AlexNet.eval()
        accuracyTotal = 0
        with torch.no_grad():
            for images, labels in trainLoader:
                images = images.to(device)
                labels = labels.to(device)
                predictions = caltech101AlexNet(images)
                accuracy = (torch.argmax(predictions, dim=1) == labels).sum()
                accuracyTotal += accuracy
        print("经{}代学习, 当前准确率为{}/{}={}".format(epoch + 25, accuracyTotal, dataAmount, accuracyTotal / dataAmount))
