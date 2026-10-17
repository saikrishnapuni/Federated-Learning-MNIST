import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import Subset,DataLoader
import copy
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
print("Traing data loaded")
test_dataset = datasets.MNIST(
    root="/data",
    train=False,
    download=True,
    transform=transform
)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False) 
def exclude_data(data,label):
    index = [idx for idx in range(len(data)) if(data[idx][1]) not in label]
    return Subset(data,indices=index)

def include_data(data,label):
    index = [idx for idx in range(len(data)) if(data[idx][1])  in label]
    return Subset(data,indices=index)

data1 = exclude_data(train_dataset,[0,3,1])
data2=exclude_data(train_dataset,[2,5,6])
data3 = exclude_data(train_dataset,[8,9,7])
client_sizes = [
    len(data1),
    len(data2),
    len(data3)
]
train_data1_loader = DataLoader(data1,batch_size=32,shuffle=True) 
train_data2_loader = DataLoader(data2,batch_size=32,shuffle=True)
train_data3_loader  = DataLoader(data3,batch_size=32,shuffle=True)

test_data1 = include_data(test_dataset,[0,3,1])
test_data2=include_data(test_dataset,[2,5,6])
test_data3 = include_data(test_dataset,[8,9,7])

test_data1_loader = DataLoader(test_data1,batch_size=32,shuffle=False) 
test_data2_loader = DataLoader(test_data2,batch_size=32,shuffle=False)
test_data3_loader  = DataLoader(test_data3,batch_size=32,shuffle=False)
 
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

global_model = Model()

local1 = Model()
local2 = Model()
local3 = Model()

def train_model(model,data):
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = 0.001)
    loss_sum = 0
    correct = 0
    total = 0
    model.train()
    for images,labels in data:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        loss_sum+=loss.item()*images.size(0)
        _,predicted = outputs.max(1)
        correct+=predicted.eq(labels).sum().item()
        total += images.size(0)
    print("------Model Training----------")
    print("Loss = ",loss_sum/total,"|"," Accuracy = ",correct*100/total)
def test_model(model,data):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for images,labels in data:
            outputs = model(images)
            _,predicted = outputs.max(1)
            correct+=predicted.eq(labels).sum().item()
            total += images.size(0)
    
    print("------Model Testing----------")
    print(" Accuracy = ",correct*100/total)     
    
for _ in range(10):
    local1.load_state_dict(global_model.state_dict())
    local2.load_state_dict(global_model.state_dict())
    local3.load_state_dict(global_model.state_dict())
    train_model(local1,train_data1_loader)
    train_model(local2,train_data2_loader)
    train_model(local3,train_data3_loader)
    w1 = local1.state_dict()
    w2 = local2.state_dict()
    w3 = local3.state_dict()
    avg_weights = copy.deepcopy(w1)
    for key in avg_weights:
        if torch.is_floating_point(avg_weights[key]):
            avg_weights[key] = (
                w1[key] * client_sizes[0]
                + w2[key] * client_sizes[1]
                + w3[key] * client_sizes[2]
            ) / sum(client_sizes)
        else:
            avg_weights[key] = w1[key]
    global_model.load_state_dict(avg_weights)


test_model(local1,test_data1_loader)
test_model(local2,test_data2_loader)
test_model(local3,test_data3_loader)


test_model(global_model,test_loader )

torch.save(global_model.state_dict(), "federated_mnist_cnn.pth")