from model import resnet

model = resnet.ResNet(resnet._Bottleneck,[3,4,6,3],seon=False)
print(model)
