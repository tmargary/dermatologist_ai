# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:39:26 2021

@author: tigra
"""

import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import model as md
import train_test as tt
import torch.optim as optim

class_names = tt.train_data.classes
print(f'number of classes: {len(class_names)}')
print(class_names)

dataiter = iter(tt.train_loader)
images, labels = dataiter.next()
images = images.numpy()  

print(images[0].shape)

from torchsummary import summary
summary(md.model_scratch, (3, 250, 250))

# loss function
criterion_scratch = nn.CrossEntropyLoss()

# optimizer
optimizer_scratch = optim.SGD(md.model_scratch.parameters(), lr=0.01)

model_scratch = tt.train(15, tt.loaders_scratch, md.model_scratch, optimizer_scratch, criterion_scratch, md.use_cuda, 'model_scratch.pt')
tt.test(tt.loaders_scratch, md.model_scratch, criterion_scratch, md.use_cuda)

# Creating the CSV
path = torch.load('C://Users/tigra/Desktop/dermatologist_ai/model_scratch.pt')
model_scratch.load_state_dict(path)
model_scratch.to('cuda')
pred = np.array([], dtype=np.float).reshape(0,3)
for batch_idx, (data, target) in enumerate(tt.loaders_scratch['test']):
    if md.use_cuda:
        data, target = data.cuda(), target.cuda()
    output = model_scratch(data)
    new = (output.cpu().detach().numpy())
    pred = np.vstack([pred, new])

predictions = np.argmax(pred, axis=1)
# task_1 list
predictions_melanoma = 1 * (predictions == 0)
# task_2 list
predictions_keratosis = 1 * (predictions == 2)

from sklearn.datasets import load_files   
data_path = "C://Users/tigra/Desktop/data/test"
data = load_files(data_path)
img_files = np.array(data['filenames'])
# Id
id_ = []
for filename in range(len(img_files)):
    id_.append(img_files[filename][34:])
    
# The list of lists (id, task_1, task_2)
csv_list = [id_, list(predictions_melanoma), list(predictions_keratosis)]

import csv
with open('my_predictions.csv', 'w', newline='') as csvfile:
    result_writger = csv.writer(csvfile)
    result_writger.writerow(['Id', 'task_1', 'task_2'])
    for i in range(len(img_files)):
        result_writger.writerow([id_[i], predictions_melanoma[i], predictions_keratosis[i]])