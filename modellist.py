from model import VGG, Resnet, DenseNet

class Modellist():
    def __init__(self):
        print('model number list')
        print('-'*30)
        print('1: Vgg11')
        print('2: Vgg13')
        print('3: Vgg16')
        print('4: Vgg19')
        print('5: Resnet18')
        print('6: Resnet34')
        print('7: Resnet50')
        print('8: Resnet101')
        print('9: Resnet152')
        print('10:DenseNet121')
        print('11:DenseNet169')
        print('12:DenseNet201')
        print('13:DenseNet161(growth_rate = 48)')
        print('-'*30)

    def __call__(self,x,seon):
        return {1: Vgg.Vgg11(seon),
        2: Vgg.Vgg13(seon),
        3: Vgg.Vgg16(seon),
        4: Vgg.Vgg19(seon),
        5: Resnet.Resnet18(seon),
        6: Resnet.Resnet34(seon),
        7: Resnet.Resnet50(seon),
        8: Resnet.Resnet101(seon),
        9: Resnet.Resnet152(seon),
        10: DenseNet121(seon),
        11: DenseNet169(seon),
        12: DenseNet201(seon),
        13: DenseNet161(seon)}[x]
