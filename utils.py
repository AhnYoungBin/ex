import os
import matplotlib.pyplot as plt
import torch

class Tester():
    def __init__(self,model,criterion,optimizer,test_loader):
        self.best_acc = 0
        self.al_dict ={'train': {'acc' : [],'loss':[]},'val': {'acc' : [],'loss':[]}}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.test_loader = test_loader

    def test(self):
        self.model.eval()
        test_loss = 0
        total = 0
        correct = 0 
        for idx, (inputs, labels) in enumerate(self.test_loader):
            inputs = inputs.to(self.device, dtype=torch.float)
            labels = labels.to(self.device, dtype=torch.long)
            with torch.no_grad():
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                loss = self.criterion(outputs, labels.squeeze(-1))
                test_loss += loss.data.cpu().numpy()

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = test_loss /len(test_data_loader)
        epoch_acc = correct / total

        print('-'*30)
        print('>> test | Loss : {:.4f}  Acc : {:.4f}'.format(epoch_loss,epoch_acc))
        print('-'*30)

    def confusion_matrix(self):
        pass






def train_graph(epoch,history_dict):
    # makefolder
    try:
        if not os.path.exists('/content/ex/workspace'):
            os.makedirs('/content/ex/workspace')
    except OSError:
        print('Error Creating director')
    
    epochs_range = range(1,epoch+1)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_dict['train']['acc'], label='Training Accuracy')
    plt.plot(epochs_range, history_dict['val']['acc'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_dict['train']['loss'], label='Training Loss')
    plt.plot(epochs_range, history_dict['val']['loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('/content/ex/workspace/train_val_graph.png', dpi=150)
    print('')
    print('The train graph is saved...')
    print('')
