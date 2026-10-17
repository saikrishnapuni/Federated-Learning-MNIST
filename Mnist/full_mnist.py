import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Subset,DataLoader
transform = transforms.Compose([
    
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


train_dataset = datasets.MNIST(
    root="/data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="/data",
    train=False,
    download=True,
    transform=transform
)
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False) 

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1,32,2,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.block2 = nn.Sequential(
                    nn.Conv2d(32,16,2,stride=2,padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(16)
                )
        self.mlp= nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*8*8,128),
            nn.ReLU(),
            nn.Linear(128,10)
            
        )
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.mlp(x)
        return x
model = Model()
epochs = 10
loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)
print("______________________________________")
print("Training Start")
for _ in range(epochs):
    model.train()
    loss_sum = 0
    total = 0
    correct = 0
    for images,labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = loss_criterion(output,labels)
        loss_sum+=loss.item()* images.size(0)
        loss.backward()
        optimizer.step()
        total += len(images)
        _,predicted = output.max(1)
        correct+=predicted.eq(labels).sum().item()
    print("loss = ",loss_sum/total," ","Accuracy = ",(correct/total)*100)
    print("______________________________________")
    print("Training Done")
    
model.eval()
    
correct = 0
total = 0
with torch.no_grad():
    for images,labels in test_loader:
        output = model(images)
        _,predicted = output.max(1)
             
        correct+=predicted.eq(labels).sum().item()
        total+=len(images)
print("______________________________________")
    
print("Testing Accuracy = ",correct*100/total)
print("______________________________________")