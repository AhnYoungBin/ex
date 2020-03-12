from model import resnet
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import data
import argparse
import modellist

PATH = '/content/gdrive/My Drive/dataset'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

modellist = modellist.Modellist()

parser = argparse.ArgumentParser(description='Learn by Modeling Dog Cat DataSet')
parser.add_argument('modelnum',type=int, help='Select your model number')
parser.add_argument('lr',type=int, help='Select opimizer learning rate')
args = parser.parse_args()

li = data.FilePath(PATH)

train_transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


Train = data.cd_Dataset(transform=train_transform, FILE_PATHS= li[0])
Val = data.cd_Dataset(transform=test_transform, FILE_PATHS= li[1])
Test = data.cd_Dataset(transform=test_transform, FILE_PATHS= li[2])

train_data_loader = DataLoader(Train, batch_size = 16, shuffle=True, num_workers=2)
val_data_loader = DataLoader(Val, batch_size = 16, shuffle=False, num_workers=2)
test_data_loader = DataLoader(Test, batch_size = 16, shuffle=False, num_workers=2)

model = resnet.ResNet(resnet._Bottleneck,[3,4,6,3],seon=False)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

best_acc = 0

def train():
    model.train()
    train_loss = 0
    total = 0
    correct = 0 

    for idx, (inputs, labels) in enumerate(train_data_loader):
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, predicted = outputs.max(1)
        loss = criterion(outputs, labels.squeeze(-1))
        loss.backward()

        train_loss += loss.data.cpu().numpy()
        optimizer.step()
        total +=labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = train_loss /len(train_data_loader)
    epoch_acc = correct/ total
    print('>> train | Loss : {.4f}  Acc : {.4f}'.format(epoch_loss,epoch_acc))
def val():
    model.eval()
    val_loss = 0
    total = 0
    correct = 0
    for idx, (inputs, labels) in enumerate(val_data_loader):
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            loss = criterion(outputs, labels.squeeze(-1))
            val_loss += loss.data.cpu().numpy()

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    epoch_loss = val_loss /len(val_data_loader)
    epoch_acc = correct / total
    print('>> test | Loss : {.4f}  Acc : {.4f}'.format(epoch_loss,epoch_acc))

epochs = 3
for epoch in range(1,epochs+1):
    print('Epoch {}/{}'.format(epoch, epochs))
    print('-'*20)
    train()
    val()

