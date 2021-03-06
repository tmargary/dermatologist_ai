# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 14:09:31 2021

@author: tigra
"""

import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import random
from PIL import ImageFile
from torchsampler import ImbalancedDatasetSampler
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("setting rendom seeds")
torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

data_path = "C://Users/tigra/Desktop/data"

print("loading and preprocessing data...")
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
valid_data = datasets.ImageFolder(data_path + "/valid", transform=preprocess['valid'])
test_data = datasets.ImageFolder(data_path + "/test", transform=preprocess['test'])

# def get_images(dataset):
#     print("getting img_dict...")
#     dataset_size = len(dataset)
#     classes = dataset.classes
#     num_classes = len(dataset.classes)
#     img_dict = {}
#     for i in range(num_classes):
#         img_dict[classes[i]] = 0
    
#     for i in range(dataset_size):
#         img, label = dataset[i]
#         img_dict[classes[label]] += 1
    
#     return img_dict

# img_dict = get_images(train_data)
# class_counts = list(img_dict.values())

# print("sampling training set...")
# num_samples = sum(img_dict.values())
# class_weights = [sum(img_dict.values())/class_counts[i] for i in range(len(list(img_dict.values())))]
# labels = []
# for batch_idx, (data, target) in enumerate(torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)):
#     labels.extend(target.cpu().detach().numpy())
# weights = [class_weights[labels[i]] for i in range(int(num_samples))]
# sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

# # choose the training and test datasets
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = sampler, num_workers=num_workers)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=ImbalancedDatasetSampler(train_data), num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

loaders_scratch = {
    'train': train_loader,
    'valid': valid_loader,
    'test': test_loader
}

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    print(f"Training model...")
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders_scratch['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))            
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
    
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
        
        # save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            valid_loss_min = valid_loss
            
    # return trained model
    return model


def test(loaders, model, criterion, use_cuda):
    print(f"Testing model...")
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        test_loss = test_loss/len(test_loader.dataset)
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))