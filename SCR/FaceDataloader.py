import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms as torch_transformations
from torch.utils.data import DataLoader, Dataset
import cv2
import albumentations as A
from os import listdir


class FaceDataset(Dataset):
    def __init__(self, image_path, annotation_path, mean, std,  aug_transform=None):
        """
        Init method for FaceDataset class, used to instantiate a FaceDataset
        :param image_path: path to image files(PNG OR JPG)
        :param annotation_path: path to annotation files(Specify folder)
        :param mean: mean/avg value of the images(1x3 list for RGB images)
        :param std: Standard deviation value of the images(1x3 list for RGB images)
        :param aug_transform: Data augmentations(albumentations type)
        :return: Nothing
        """
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.mean = mean
        self.std = std
        self.aug_transform = aug_transform  # augmentation transforms
        self.file_list = os.listdir(self.image_path)

    def __len__(self):
        """
        Returns the size of the datasets i.e. the number of data samples in the dataset
        :return: Returns the size of the datasets
        """
        return len(listdir(self.image_path))  # Compute the dataset length by counting the number of files indexed

    def __getitem__(self, index):
        """
        Used by the dataloader function to load in the data
        :param index: provided by the Dataloader to iterate over the data
        :return: image, label, arousal, valence(in this order)
        """
        image = cv2.imread(self.image_path + "/" + self.file_list[
            index])  # Index over the indexed list of files to account over any labeling errors
        file_num = self.file_list[index].split("/")
        file_num = file_num[-1].split(".")[0]  # Get the label of the files(file name)

        label = np.load(self.annotation_path + "/" + file_num + "_exp.npy")
        arousal = np.load(self.annotation_path + "/" + file_num + "_aro.npy")
        valence = np.load(self.annotation_path + "/" + file_num + "_val.npy")

        if self.aug_transform is not None:
            aug = self.aug_transform(image=image)
            image = Image.fromarray(aug["image"])

        transformations = torch_transformations.Compose(
            [torch_transformations.ToTensor(), torch_transformations.Normalize(self.mean, self.std)])
        if self.aug_transform is None:
            image = Image.fromarray(image)

        image = transformations(image) # Normalization of the images
        label = torch.tensor(int(label))
        arousal = torch.tensor(float(arousal))
        valence = torch.tensor(float(valence))

        return image, label, arousal, valence
