import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Caltech101
import cv2

trainTransform = torchvision.transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
     ])
class Caltech101Dataset(Caltech101):
    def __init__(self, root, transform=None):
        super(Caltech101Dataset, self).__init__(root, transform=transform)
        self.root = root
        self.transform = transform

    def __getitem__(self,index):
        image,tartget = Caltech101.__getitem__(self, index)
        if image.shape[0]!=3:
            image = torch.concat([image,image,image], dim=0)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(image)
        return image,tartget


caltech101Dataset = Caltech101Dataset("D:\\WorkSpace\\DataSet\\Caltech", transform=trainTransform)
trainLoader = DataLoader(caltech101Dataset, batch_size=64, shuffle=True)
testLoader = DataLoader(caltech101Dataset, batch_size=64, shuffle=True)
# print(caltech101Dataset.annotation_categories)
trainSlice = slice(7000)
testSlice = slice(1677)
trainDataset = caltech101Dataset
testDataset = caltech101Dataset
# print(testDataset)

