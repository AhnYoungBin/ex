import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import data
import argparse
import modellist
import matplotlib.pyplot as plt

PATH = '/content/gdrive/My Drive/dataset'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

modellist = modellist.Modellist()

parser = argparse.ArgumentParser(description='Learn by Modeling Dog Cat DataSet')
parser.add_argument('modelnum',type=int, help='Select your model number')
parser.add_argument("-se", help="Put the selayer in the model.",
                    action="store_true")
parser.add_argument("-show", help="show to model Archtecture",
                    action="store_true")

parser.add_argument('lr',type=float, help='Select opimizer learning rate')
parser.add_argument('epochs',type=int, help='Select train epochs')

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

model = modellist(args.modelnum, seon = args.se)
if args.show:
    print(model)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

best_acc = 0
train_loss_list =[]
val_loss_list = []
train_acc_list = []
val_acc_list = []
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

    train_loss_list.append(epoch_loss)
    train_acc_list.append(epoch_acc)

    print('>> train | Loss : {:.4f}  Acc : {:.4f}'.format(epoch_loss,epoch_acc))
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

    val_loss_list.append(epoch_loss)
    val_acc_list.append(epoch_acc)

    print('>> test | Loss : {:.4f}  Acc : {:.4f}'.format(epoch_loss,epoch_acc))

for epoch in range(1,args.epochs+1):
    print('Epoch {}/{}'.format(epoch, args.epochs))
    print('-'*20)
    train()
    val()


epochs_range = range(args.epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc_list, label='Training Accuracy')
plt.plot(epochs_range, val_acc_list, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss_list, label='Training Loss')
plt.plot(epochs_range, val_loss_list, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


test_loss_list = []
test_acc_list = []
predicted_list = []
label_list = []
def test():
    model.eval()
    test_loss = 0
    total = 0
    correct = 0
    for idx, (inputs, labels) in enumerate(test_data_loader):
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            loss = criterion(outputs, labels.squeeze(-1))
            test_loss += loss.data.cpu().numpy()

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            predicted_list.append(predicted)
            label_list.append(labels)

    epoch_loss = test_loss /len(test_data_loader)
    epoch_acc = correct / total

    test_loss_list.append(epoch_loss)
    test_acc_list.append(epoch_acc)
    print('-'*30)
    print('>> test | Loss : {:.4f}  Acc : {:.4f}'.format(epoch_loss,epoch_acc))

test()

