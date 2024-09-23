import os
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as dataset
import numpy as np
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from custom_rn import resnet18

resnet18 = resnet18()
resnet18.fc = nn.Linear(512,17)
resnet18 = resnet18.cuda()
data_transform = {
    "train": transforms.Compose([transforms.Resize((256, 256)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                                 ]),
    }

train_dataset = dataset.ImageFolder(root="/media/26d532/gxx/PADISI_USC/pasidi_attack/", transform=data_transform["train"])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=6,drop_last = True) 
optimizer = optim.SGD(resnet18.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-3)
criteria = nn.CrossEntropyLoss()
num_epochs = 200
for epoch in range(0,num_epochs):
    print("Epoch: {}/{}".format(epoch + 1, num_epochs))
    running_loss = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].cuda(), data[1].cuda()
        loss_asym = 0
        optimizer.zero_grad()
        _,_,_,outputs = resnet18(inputs) 
        loss_asym = criteria(outputs, labels) 
        loss = loss_asym
        loss.backward()
        optimizer.step()        
        running_loss += loss.item() * inputs.size(0)
        rus = running_loss / len(train_dataset)
    print("\t Training: Loss: {:.4f}".format(rus))
    torch.save(resnet18, './PADISI.pkl')   