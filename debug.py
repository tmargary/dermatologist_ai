dataset = train_data
dataset_size = len(dataset)
classes = dataset.classes
num_classes = len(dataset.classes)
img_dict = {}
for i in range(num_classes):
    img_dict[classes[i]] = 0

for i in range(dataset_size):
    img, label = dataset[i]
    img_dict[classes[label]] += 1

img_dict