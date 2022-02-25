import torch
import torchvision
from  torch import nn
from torchvision.models import alexnet, AlexNet
from Caltech101Dataset import caltech101Dataset, trainLoader
import time

class Caltech101AlexNet2(nn.Module):
    def __init__(self, preTrain=False):
        super(Caltech101AlexNet2, self).__init__()
        if preTrain:
            self.alexNetLayer = alexnet(pretrained=True)

            self.alexNet = self.alexNetLayer
        else:
            self.alexNet = AlexNet(num_classes=101)

    def forward(self, x):
        x = self.alexNet(x)
        return x

if __name__ == "__main__":
    # caltech101AlexNet = Caltech101AlexNet2(preTrain=True)
    caltech101AlexNet = Caltech101AlexNet2(preTrain=False)
    print(caltech101AlexNet)
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    dataAmount = caltech101Dataset.__len__()
    maxEpoch = 10
    # caltech101AlexNet = Caltech101AlexNet(preTrain=True)
    caltech101AlexNet = torch.load("./model/AlexNet2.pth")
    caltech101AlexNet.to(device)
    # print(caltech101AlexNet)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(caltech101AlexNet.parameters(), lr=1e-4, momentum=0.9)
    steps = 0
    for epoch in range(maxEpoch):
        startTime = time.time()
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
        print("经{}代学习, 当前损失值为{},用时:{}".format(epoch + 11, lossEpoch, endTime - startTime))
        torch.save(caltech101AlexNet, "./model/AlexNet2.pth")
        caltech101AlexNet.eval()
        accuracyTotal = 0
        with torch.no_grad():
            for images, labels in trainLoader:
                images = images.to(device)
                labels = labels.to(device)
                predictions = caltech101AlexNet(images)
                accuracy = (torch.argmax(predictions, dim=1) == labels).sum()
                accuracyTotal += accuracy
        print("经{}代学习, 当前准确率为{}/{}={}".format(epoch + 11, accuracyTotal, dataAmount, accuracyTotal / dataAmount))
