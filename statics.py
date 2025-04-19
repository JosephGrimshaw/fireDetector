from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from matplotlib import pyplot as plt
import random
import torch

train_csv = './data/fireTrainData.csv'
test_csv = './data/fireTestData.csv'

train_filedir = './data/trainData/'
test_filedir = './data/testData/'

class Dataset(Dataset):
    def __init__(self, csv, folder_dir, transform=None):
        self.csv_file = csv
        self.folder_dir = folder_dir
        self.data_labels = pd.read_csv(csv)
        self.transform = transform
        self.len = self.data_labels.shape[0]

    def __getitem__(self, idx):
        img_name = self.data_labels.iloc[idx, 0]
        label = self.data_labels.iloc[idx, 1]
        path = self.folder_dir + img_name
        img = Image.open(path)
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return self.len

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize(mean, std)])

train_dataset = Dataset(train_csv, train_filedir, transform=composed)
test_dataset = Dataset(test_csv, test_filedir, transform=composed)

train_loader = DataLoader(train_dataset, batch_size=36, shuffle=True)
test_loader = DataLoader(test_dataset)

'''
trainNumbers = random.sample(range(1, 5000), 10)
testNumbers = random.sample(range(1, 25), 10)
for i in trainNumbers:
    img, label = train_dataset.__getitem__(i)
    print("Class: ", label)
    plt.imshow(img)
    plt.show()

for i in testNumbers:
    img, label = test_dataset.__getitem__(i)
    print("Class: ", label)
    plt.imshow(img)
    plt.show()
'''

