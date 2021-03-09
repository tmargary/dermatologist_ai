# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:06:15 2021

@author: tigra
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# define the CNN architecture
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
        self.fc2 = nn.Linear(500, 133) # 500 flatens to 133 (number of output class)
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

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
    
    
    
