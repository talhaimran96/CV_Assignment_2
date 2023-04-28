import numpy as np
import torch
import torchvision.models
from torchvision.transforms import transforms as torch_transformations
from FaceDataloader import FaceDataset
from torch.utils.data import DataLoader
import cv2
import albumentations as A
import os
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import CustomVGGModel
from torchsummary import summary

# these values come from imagenet, i.e leverage the relationships of natural objects
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

image_path = "../Dataset/val_set/images"
annotation_path = "../Dataset/val_set/annotations"

t_val = A.Compose([A.HorizontalFlip(),
                   A.GridDistortion(p=0.2)])

test_set = FaceDataset(image_path, annotation_path, mean, std, t_val)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

list_of_files = os.listdir(image_path)

# img, _, _, _ = test_set.__getitem__(10)
# print((img))
# # cv2.imshow(" ", img.transpose((1,2,0)))
# # cv2.waitKey(0)
#
#
# for image_batch, labels, Aro, Val in test_loader:
#     print(image_batch, labels, Aro, Val)


train_accuracy_list = []
val_accuracy_list = []

train_loss_list = []
val_loss_list = []

val_accuracy_max = -1

# plt.figure()
# plt.plot(train_accuracy_list, label="train_accuracy")
# plt.plot(val_accuracy_list, label="val_accuracy")
# plt.legend()
#
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
#
# plt.title('Training and val Accuracy')
# plt.savefig('Training and val Accuracy')
#
# plt.figure()
# plt.plot(train_loss_list, label="train_loss")
# plt.plot(val_loss_list, label="val_loss")
#
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and val Loss')
# plt.savefig('Training and val Loss')

# temp = torchvision.models.vgg16_bn()
# temp_1 = temp.parameters()
# layers = [x.data for x in temp.parameters()]
# temp_1.req
# print("Correlation Arousal: ", np.corrcoef([1, 2, 2, 3], [1, 2, 3, 4]))
#
# print(average_precision_score(np.array([1, 1, 1, 0, 0, 0, 1, 1, 1]), np.array([1, 1, 0, 0, 0, 1, 1, 1, 0])))


from sklearn.utils import class_weight
import pandas as pd
import numpy as np

y_train = pd.DataFrame(['dog', 'dog', 'dog', 'dog', 'dog',
                        'cat', 'cat', 'cat', 'bat', 'bat'])

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train.values.reshape(-1)
)

print(weights,np.unique(y_train),y_train.values.reshape(-1))