import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import torchvision

from functools import partial
from matplotlib import pyplot as plt
from random import randrange
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import HyperBandForBOHB, ASHAScheduler
from tabulate import tabulate
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
from torch.utils.data import TensorDataset
import random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# standard cast into Tensor and pixel values normalization in [-1, 1] range
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# extra transfrom for the training data, in order to achieve better performance
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
    torchvision.transforms.RandomHorizontalFlip(), 
])

dec_x_train = np.loadtxt('1_trn_img.txt')
dec_y_train = np.loadtxt('1_trn_lab.txt')
dec_x_train = dec_x_train.reshape(dec_x_train.shape[0],3,32,32)

dec_x_test = np.loadtxt('0_tst_img.txt') 
dec_y_test = np.loadtxt('0_tst_lab.txt')
dec_x_test = dec_x_test.reshape(dec_x_test.shape[0],3,32,32)

tensor_x = torch.Tensor(dec_x_train)
tensor_y = torch.tensor(dec_y_train,dtype=torch.long)
trainset = TensorDataset(tensor_x, tensor_y)

tensor_x = torch.Tensor(dec_x_train)
tensor_y = torch.tensor(dec_y_train,dtype=torch.long)
validationset = TensorDataset(tensor_x, tensor_y)

tensor_x = torch.Tensor(dec_x_test)
tensor_y = torch.tensor(dec_y_test,dtype=torch.long)
testset = TensorDataset(tensor_x, tensor_y)

'''
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform
)
validationset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

print('Training set')
print(f'Samples: {trainset.data.shape}')
print(f'Labels: {len(trainset.targets)}')

print('\nTest set')
print(f'Samples: {testset.data.shape}')
print(f'Labels: {len(testset.targets)}')

print('\nClasses\n')
print(tabulate(
    list(trainset.class_to_idx.items()), headers=['Name', 'Index'], 
    tablefmt='orgtbl'
))
'''

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.res1 = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.res2 = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        ), nn.Sequential( 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4), 
            nn.Flatten(), 
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        
        return x

# percentage of training set to use as validation
valid_size = 0.2

# obtain training indices that will be used for validation
num_train = len(trainset)
indices = list(range(num_train))
random.shuffle(indices)

split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
TRAIN_SAMPLER = SubsetRandomSampler(train_idx)
VALID_SAMPLER = SubsetRandomSampler(valid_idx)

# number of subprocesses to use for data loading
NUM_WORKERS = 32

def data_loaders(trainset, validationset, testset, size):
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=size, 
        sampler=TRAIN_SAMPLER, num_workers=NUM_WORKERS
    )
    validloader = torch.utils.data.DataLoader(
        validationset, batch_size=size, 
        sampler=VALID_SAMPLER, num_workers=NUM_WORKERS
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=size, num_workers=NUM_WORKERS
    )

    return trainloader, validloader, testloader

def train_cifar(
    config, trainset, validationset, testset, 
    epochs=10, checkpoint_dir=None, tuning=False
):
    net = ResNet()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        net.parameters(), 
        lr=config['lr'], 
        betas=(config['beta1'], config['beta2']), 
        amsgrad=config['amsgrad'], 
    )

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, 'checkpoint')
        model_state, optimizer_state = torch.load(checkpoint)
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainloader, validloader, testloader = data_loaders(
        trainset, validationset, testset, config['batch_size']
    )

    train_loss_list = []
    accuracy_list = []

    # track minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(epochs):
        
        train_loss = 0.0
        net.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        correct = 0
        valid_loss = 0.0
        net.eval()
        for inputs, labels in validloader:
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                predicted = torch.max(outputs.data, 1)[1]

                correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(TRAIN_SAMPLER)
        valid_loss = valid_loss / len(VALID_SAMPLER)
        accuracy = correct / len(VALID_SAMPLER)

        train_loss_list.append(train_loss)
        accuracy_list.append(accuracy)

        if not tuning:
            print(
                f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \t'
                f'Validation Loss: {valid_loss:.6f} \t'
                f'Validation Accuracy: {accuracy:.6f}'
            )

            if valid_loss <= valid_loss_min:
                print(
                    'Validation loss decreased ('
                    f'{valid_loss_min:.6f} --> {valid_loss:.6f}).  '
                    'Saving model ...'
                )

                torch.save(net.state_dict(), 'cnn.pt')
                valid_loss_min = valid_loss
        else:
            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and will potentially be passed as the `checkpoint_dir`
            # parameter in future iterations.
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'checkpoint')
                torch.save((net.state_dict(), optimizer.state_dict()), path)

            tune.report(mean_loss=valid_loss, accuracy=accuracy)

    print(f'\n----------- Finished Training -----------')

    return train_loss_list, accuracy_list

def test_accuracy(net, testloader):
    correct = 0
    pred_fin = []
    target_fin = []
    output_fin = np.empty((0,10))
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        net.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            # calculate outputs by running images through the network
            outputs = net(images)

            # the class with the highest energy is what we choose as prediction
            predicted = torch.max(outputs.data, 1)[1]

            correct += (predicted == labels).sum().item()

            outputs = torch.exp(outputs)
            pred_fin = np.append(pred_fin, torch.Tensor.cpu(predicted))
            target_fin = np.append(target_fin, torch.Tensor.cpu(labels))
            output_fin = np.append(output_fin, torch.Tensor.cpu(outputs), axis=0)
    
    return correct / len(testloader.dataset), target_fin, pred_fin, output_fin
    
    
def test_accuracy_per_class(net, testloader):
    correct_pred = {classname: 0 for classname in trainset.classes}
    total_pred = {classname: 0 for classname in trainset.classes}

    with torch.no_grad():
        net.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            predicted = torch.max(outputs.data, 1)[1]

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[trainset.classes[label]] += 1
                total_pred[trainset.classes[label]] += 1
    
    accuracy_per_class = {classname: 0 for classname in trainset.classes}
    for classname, correct_count in correct_pred.items():
        accuracy = (100 * float(correct_count)) / total_pred[classname]
        accuracy_per_class[classname] = accuracy

    return accuracy_per_class

EPOCHS = 20

config = {                                                                                                                                                                                                          
    'batch_size': 16,
    'lr': 8.0505e-05,
    'beta1': 0.851436,
    'beta2': 0.999689,
    'amsgrad': True
} 

train_loss_list, accuracy_list = train_cifar(
    config, trainset, validationset, testset, epochs=EPOCHS
)

_, _, testloader = data_loaders(
    trainset, validationset, testset, config['batch_size']
)

trained_net = ResNet()
trained_net.to(device)
trained_net.load_state_dict(torch.load('cnn.pt'))

overall_accuracy, target, pred, output = test_accuracy(trained_net, testloader)

print(
    'Overall accuracy of the network  '
    f'{(overall_accuracy * 100):.2f} %\n'
    'on the 10000 test images'
)

array = confusion_matrix(target, pred)
from imblearn.metrics import geometric_mean_score

print("The geometric mean:", "{:.3f}".format(geometric_mean_score(target, pred)))

from imblearn.metrics import make_index_balanced_accuracy
alpha = 0.1
geo_mean = make_index_balanced_accuracy(alpha=alpha, squared=True)(geometric_mean_score)

print("The IBA using alpha=" + str(alpha) + " and the geometric mean:", "{:.3f}".format(geo_mean(target, pred)))

from sklearn.metrics import roc_auc_score

print("The AUROC:", "{:.3f}".format(roc_auc_score(target, output, multi_class='ovr')))
   
cm = pd.DataFrame(array, index = range(10), columns = range(10))
plt.figure(figsize=(20,20))
sns.heatmap(cm, annot=True)
plt.show()

'''
accuracy_per_class = test_accuracy_per_class(trained_net, testloader)

print('Accuracy per class\n')
for classname, accuracy in accuracy_per_class.items():
    print(f'{classname:12s} {accuracy:.2f} %')
'''
