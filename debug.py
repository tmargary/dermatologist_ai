# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:50:19 2021

@author: tigra
"""

import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms





# check if CUDA is available
use_cuda = torch.cuda.is_available()

data_path = "C://Users/tigra/Desktop/data"

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

# convert data to a normalized torch.FloatTensor
preprocess = {
                'train' : transforms.Compose([transforms.RandomResizedCrop(250),
                                              transforms.RandomRotation(30),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),

                'valid' : transforms.Compose([transforms.CenterCrop(250),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),

                'test' : transforms.Compose([transforms.CenterCrop(250),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])}

# choose the training, test, and validation datasets
train_data = datasets.ImageFolder(data_path + "/train", transform=preprocess['train'])
valid_data = datasets.ImageFolder(data_path + "/test", transform=preprocess['valid'])
test_data = datasets.ImageFolder(data_path + "/valid", transform=preprocess['test'])

# choose the training and test datasets
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)


loaders_scratch = {
    'train': train_loader,
    'valid': valid_loader,
    'test': test_loader
}
    
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        # convolutional layer (sees 250x250x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) # ( (W-K+2P)/S ) + 1
        # convolutional layer (sees 125x125x16 image tensor) maxpooled
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 62x62x32 image tensor) maxpooled
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1) 
        # convolutional layer (sees 31x31x64 image tensor) maxpooled
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1) 
        # convolutional layer (sees 15x15x128 image tensor) maxpooled
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1) #-> 7x7x256
        
        self.conv_bachn1 = nn.BatchNorm2d(16)
        self.conv_bachn4 = nn.BatchNorm2d(128)
    
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (256 * 7 * 7 -> 500)
        self.fc1 = nn.Linear(256 * 7 * 7, 500) # 256 * 7 * 7 flatens to 500 (arbitrary number)
        self.fc2 = nn.Linear(500, 3) # 500 flatens to 133 (number of output class)
        self.dropout = nn.Dropout(0.30)
    
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv_bachn1(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv_bachn4(self.conv4(x))))
        x = self.pool(F.relu(self.conv5(x)))
        # flatten image input
        x = x.view(-1, 256 * 7 * 7)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net() 
    
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
    
test_loss = 0.
correct = 0.
total = 0.


path = torch.load('C://Users/tigra/Desktop/dermatologist_ai/model_scratch.pt')
model_scratch.load_state_dict(path)
model_scratch.to('cuda')

dataiter = iter(test_loader)
data, target = dataiter.next()
data, target = data.cuda(), target.cuda()
output = model_scratch(data)
print(output)

model_scratch.load_weights(path)
import csv






my_predictions = []
for batch_idx, (data, target) in enumerate(loaders_scratch['test']):
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    output = model_scratch(data)








for batch_idx, (data, target) in enumerate(loaders_scratch['test']):
    # move to GPU
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model_scratch(data)
    # calculate the loss
    loss = criterion(output, target)
    # update average test loss 
    test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
    test_loss = test_loss/len(test_loader.dataset)
    # convert output probabilities to predicted class
    pred = output.data.max(1, keepdim=True)[1]
    #print(batch_idx)
    print(pred)
    # compare predictions to true label
    correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
    print(correct)
    total += data.size(0)