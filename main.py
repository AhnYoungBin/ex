from model import resnet
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import data

PATH = '/content/gdrive/My Drive/dataset'

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

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

