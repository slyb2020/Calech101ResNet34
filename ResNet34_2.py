import torch
import torchvision
from torch import nn
from torchvision.models import resnet34
from torchsummary import summary
from Caltech101Dataset import caltech101Dataset, trainLoader
import time

device = torch.device("cuda")


class Clatech101ResNet34(nn.Module):
    def __init__(self, preTrain=True):
        super(Clatech101ResNet34, self).__init__()
        if preTrain:
            self.model = resnet34(pretrained=True)
            for layer in self.model.children():
                layer.requires_grad = False
            self.model.fc = nn.Linear(512, 101)
            self.model.fc.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    clatech101ResNet34 = Clatech101ResNet34()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(clatech101ResNet34.parameters(), lr=1e-3, momentum=0.9)
    datasetSize = caltech101Dataset.__len__()
    maxEpoch = 10
    caltech101AlexNet = torch.load("./model/ResNet34.pth")
    for epoch in range(maxEpoch):
        lossEpoch = 0
        startTime = time.time()
        clatech101ResNet34.train()
        for images, labels in trainLoader:
            optimizer.zero_grad()
            predictions = clatech101ResNet34(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            lossEpoch += loss
        trainTime = time.time()
        torch.save(clatech101ResNet34, "./model/ResNet34.pth")
        clatech101ResNet34.eval()
        accuracyTotal = 0
        with torch.no_grad():
            for images, labels in trainLoader:
                predictions = clatech101ResNet34(images)
                accuracy = (torch.argmax(predictions, dim=1) == labels).sum()
                accuracyTotal += accuracy
        testTime = time.time()
        print("完成第{}/{}代训练，损失:{},训练用时{}，测试用时{}，识别准确率{}/{}={}".format(epoch+1, maxEpoch, lossEpoch,
                                                                     trainTime - startTime, testTime - trainTime,
                                                                     accuracyTotal, datasetSize,
                                                                     accuracyTotal / datasetSize * 100))
# clatech101ResNet34 = resnet34(pretrained=True)
# clatech101ResNet34.to(device)
# summary(clatech101ResNet34,(3,224,224))
