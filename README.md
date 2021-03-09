# Dermatologist AI

In this project, I will design an algorithm that can visually diagnose melanoma, the deadliest form of skin cancer. In particular, the algorithm will distinguish this malignant skin tumor from two types of benign lesions (nevi and seborrheic keratoses).

The data and objective are pulled from the 2017 ISIC Challenge on [Skin Lesion Analysis Towards Melanoma Detection](https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a).

![Image of Yaktocat](https://github.com/tmargary/dermatologist_ai/blob/main/images/skin_disease_classes.png)

Classes: ['melanoma', 'nevus', 'seborrheic_keratosis']

CNN architecture:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 16, 250, 250]             448
       BatchNorm2d-2         [-1, 16, 250, 250]              32
         MaxPool2d-3         [-1, 16, 125, 125]               0
            Conv2d-4         [-1, 32, 125, 125]           4,640
         MaxPool2d-5           [-1, 32, 62, 62]               0
            Conv2d-6           [-1, 64, 62, 62]          18,496
         MaxPool2d-7           [-1, 64, 31, 31]               0
            Conv2d-8          [-1, 128, 31, 31]          73,856
       BatchNorm2d-9          [-1, 128, 31, 31]             256
        MaxPool2d-10          [-1, 128, 15, 15]               0
           Conv2d-11          [-1, 256, 15, 15]         295,168
        MaxPool2d-12            [-1, 256, 7, 7]               0
          Dropout-13                [-1, 12544]               0
           Linear-14                  [-1, 500]       6,272,500
          Dropout-15                  [-1, 500]               0
           Linear-16                  [-1, 133]          66,633
================================================================
```

