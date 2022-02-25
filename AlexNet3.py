import torch
import torchvision
from torchvision.models import alexnet
from torch import nn
from Caltech101Dataset import trainLoader, caltech101Dataset
import time
from torchsummary import summary


class Caltech101AlexNet3(nn.Module):
    def __init__(self, preTrain=False):
        super(Caltech101AlexNet3, self).__init__()
        if preTrain:
            self.features = alexnet(pretrained=True).features
            self.features.requires_grad = False
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 101)
            )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda")
    caltech101AlexNet = Caltech101AlexNet3(preTrain=True)
    caltech101AlexNet.to(device)
    summary(caltech101AlexNet, input_size=(3, 224, 224), batch_size=-1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(caltech101AlexNet.parameters(), lr=1e-4, momentum=0.9)
    maxEpoch = 1
    caltech101AlexNet = torch.load("./model/AlexNet3.pth")
    dataSize = caltech101Dataset.__len__()
    for epoch in range(maxEpoch):
        caltech101AlexNet.train()
        step = 0
        lossEpoch = 0
        startTime = time.time()
        for images, labels in trainLoader:
            optimizer.zero_grad()
            predictions = caltech101AlexNet(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            step += 1
            lossEpoch += loss
        torch.save(caltech101AlexNet, "./model/AlexNet3.pth")
        trainTime = time.time()
        caltech101AlexNet.eval()
        with torch.no_grad():
            accuracyTotal = 0
            for images, labels in trainLoader:
                predictions = caltech101AlexNet(images)
                accuracy = (torch.argmax(predictions, dim=1) == labels).sum()
                accuracyTotal += accuracy
            testTime = time.time()
        print("已完成{}/{}代训练，损失值:{}，训练用时:{}，准确率：{}/{}={}， 测试用时：{}".format(epoch+1, maxEpoch, lossEpoch,
                        trainTime - startTime, accuracyTotal, dataSize ,accuracyTotal/dataSize, testTime - trainTime))
