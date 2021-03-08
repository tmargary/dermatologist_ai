# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:39:26 2021

@author: tigra
"""

import os
import torch
import torchvision

from torch.utils.data.sampler import SubsetRandomSampler
import model as md
import training

class_names = training.train_data.classes
print(f'number of classes: {len(class_names)}')
print(class_names)

dataiter = iter(training.train_loader)
images, labels = dataiter.next()
images = images.numpy() 

print(images[0].shape)

from torchsummary import summary
summary(md.model_scratch, (3, 250, 250))

### TODO: select loss function
criterion_scratch = md.nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = md.optim.SGD(md.model_scratch.parameters(), lr=0.01)

model_scratch = training.train(10, training.loaders_scratch, md.model_scratch, optimizer_scratch, criterion_scratch, md.use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))