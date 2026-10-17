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
print(train_dataset[0][0].shape)

def exclude_data(data,label):
    index = [idx for idx in range(len(data)) if(data[idx][1]) not in label]
    return Subset(data,indices=index)

def include_data(data,label):
    index = [idx for idx in range(len(data)) if(data[idx][1])  in label]
    return Subset(data,indices=index)

data1 = exclude_data(train_dataset,[0,3,1])
data2=exclude_data(train_dataset,[2,5,6])
data3 = exclude_data(train_dataset,[8,9,7])
 
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

        
train_data1_loader = DataLoader(data1,batch_size=32,shuffle=True) 
train_data2_loader = DataLoader(data2,batch_size=32,shuffle=True)
train_data3_loader  = DataLoader(data3,batch_size=32,shuffle=True)

def train_model(model,data):
    
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
        for images,labels in data:
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
    print("Training Done")
    
    print("______________________________________")
model1 = Model()
model2 = Model()
model3 = Model()

train_model(model1,train_data1_loader)
train_model(model2,train_data2_loader)
train_model(model3,train_data3_loader)


test_data1 = include_data(test_dataset,[0,3,1])
test_data2=include_data(test_dataset,[2,5,6])
test_data3 = include_data(test_dataset,[8,9,7])

test_data1_loader = DataLoader(test_data1,batch_size=32,shuffle=False) 
test_data2_loader = DataLoader(test_data2,batch_size=32,shuffle=False)
test_data3_loader  = DataLoader(test_data3,batch_size=32,shuffle=False)

def test_model(model,data):
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images,labels in data:
            output = model(images)
            _,predicted = output.max(1)
             
            correct+=predicted.eq(labels).sum().item()
            total+=len(images)
    print("______________________________________")
    
    print("Testing Accuracy = ",correct*100/total)
    print("______________________________________")
    
test_model(model1,test_data1_loader)
test_model(model2,test_data2_loader)
test_model(model3,test_data3_loader)