# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:39:26 2021

@author: tigra
"""

import os
import torch
import torchvision
import torch.nn as nn
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
import model as md
import train_test

class_names = train_test.train_data.classes
print(f'number of classes: {len(class_names)}')
print(class_names)

dataiter = iter(train_test.train_loader)
images, labels = dataiter.next()
images = images.numpy() 

print(images[0].shape)

from torchsummary import summary
summary(md.model_scratch, (3, 250, 250))

# loss function
criterion_scratch = nn.CrossEntropyLoss()

# optimizer
optimizer_scratch = md.optim.SGD(md.model_scratch.parameters(), lr=0.01)

model_scratch = train_test.train(5, train_test.loaders_scratch, md.model_scratch, optimizer_scratch, criterion_scratch, md.use_cuda, 'model_scratch.pt')

path = torch.load('C://Users/tigra/Desktop/dermatologist_ai/model_scratch.pt')
model_scratch.load_state_dict(path)
model_scratch.to('cuda')
csv = np.array([], dtype=np.float).reshape(0,3)
for batch_idx, (data, target) in enumerate(train_test.loaders_scratch['test']):
    if md.use_cuda:
        data, target = data.cuda(), target.cuda()
    output = model_scratch(data)
    new = (output.cpu().detach().numpy())
    csv = np.vstack([csv, new])

print(csv)
    
    
    
    