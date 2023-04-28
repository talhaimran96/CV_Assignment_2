import torch
from torch import nn
import torchvision
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
from torchsummary import summary
from FaceDataloader import FaceDataset
from torch.utils.data import DataLoader
import cv2
import albumentations as A
import os
import matplotlib.pyplot as plt
import wandb

# these values come from imagenet, i.e leverage the relationships of natural objects
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_image_path = "../Dataset/val_set/images"
test_annotation_path = "../Dataset/val_set/annotations"

train_image_path = "../Dataset/train_set/images"
train_annotation_path = "../Dataset/train_set/annotations"
batch_size = 512

t_val = A.Compose([A.HorizontalFlip(),
                   A.GridDistortion(p=0.2)])

test_set1 = FaceDataset(test_image_path, test_annotation_path, mean, std, t_val)
test_loader1 = DataLoader(test_set1, batch_size=batch_size, shuffle=False)

train_set = FaceDataset(train_image_path, train_annotation_path, mean, std, t_val)

train_size = int(0.8 * len(train_set))
test_size = len(train_set) - train_size
train_set, test_set = torch.utils.data.random_split(train_set, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

print(test_size, train_size)


class Talha_net(nn.Module):
    def __init__(self):
        super(Talha_net, self).__init__()

        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)  # (input channels, output channels,------)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU(inplace=True)  ## note thr inplace, is it important????
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Defining another 2D convolution layer
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # linear(input,output)
        self.linear_layers = nn.Linear(32 * 14 * 14, 8)  ## flatten the image here. i.e linear layer!!
        self.regression_layer_1 = nn.Linear(32 * 14 * 14, 32)
        self.regression_layer_2 = nn.Linear(32, 1)
        self.LeakyRelu = nn.LeakyReLU(0.1)

    # Defining the forward pass
    def forward(self, x):
        x = self.conv1(x)
        print(type(x))
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # Apply conv2, bn2, relu2 and maxpool2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # Apply conv3, bn3, relu3 and maxpool3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        # Apply conv4, bn4, relu4 and maxpool4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        # Flatten the output from the conv layers
        x = x.view(x.size(0), -1)

        # Apply the linear layer
        y = self.regression_layer_1(x)
        arr = self.regression_layer_2(y)
        val = self.regression_layer_2(y)
        classification = self.linear_layers(x)

        return classification, arr, val


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device="cpu"
dummy_model = Talha_net().to(device)
# summary(dummy_model, (3, 224, 224))

# Select a loss function
loss_function1 = torch.nn.CrossEntropyLoss()  # applies the softmax at the output intrinsically
loss_function2 = nn.MSELoss()
# Select an optimizer
optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
# For more on schedulers please visit:
# https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863

# Set the number of epochs
# This variable is used in the main training loop
epochs = 10


def run_1_epoch(model, loss_function1, loss_function2, optimizer, loader, train=False):
    if train:
        model.train()
    else:
        model.eval()

    total_correct_preds = 0

    total_loss = 0

    # Number of images we can get by the loader
    total_samples_in_loader = (loader.dataset.__len__())

    # number of batches we can get by the loader
    total_batches_in_loader = len(loader)
    for image, label, arousal, valence in tqdm(loader):

        # Transfer image_batch to GPU if available
        image = image.to(device)
        label = label.to(device)
        arousal = arousal.to(device)
        valence = valence.to(device)

        # Zeroing out the gradients for parameters
        if train:
            optimizer.zero_grad()

        # Forward pass on the input batch
        output_class, output_arousal, output_valence = model(image)

        # Acquire predicted class indices
        _, predicted = torch.max(output_class.data, 1)  # the dimension 1 corresponds to max along the rows

        # Removing extra last dimension from output tensor
        output_class.squeeze_(-1)
        arousal = arousal.unsqueeze(1)
        valence = valence.unsqueeze(1)
        # Compute the loss for the minibatch
        loss_1 = loss_function1(output_class, label)
        loss_2 = loss_function2(output_arousal, arousal)
        loss_3 = loss_function2(output_valence, valence)

        loss = loss_1 + loss_2 + loss_3

        # Backpropagation
        if train:
            loss.backward()

        # Update the parameters using the gradients
        if train:
            optimizer.step()

        # Extra variables for calculating loss and accuracy
        # count total predictions for accuracy calcutuon for this epoch
        total_correct_preds += (predicted == label).sum().item()

        total_loss += loss.item()

    loss = total_loss / total_batches_in_loader
    accuracy = 100 * total_correct_preds / total_samples_in_loader

    return loss, loss_1, loss_2, loss_3, accuracy


# Some helper variables

train_accuracy_list = []
val_accuracy_list = []

train_loss_list = []
val_loss_list = []

val_accuracy_max = -1

wandb.init(
    # set the wandb project where this run will be logged
    project="Custom_model-test",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.001,
        "architecture": "CNN",
        "dataset": "Faces",
        "epochs": epochs,
    }
)

# Main training and validation loop for n number of epochs
for i in range(epochs):

    # Train model for one epoch

    # Get the current learning rate from the optimizer
    current_lr = optimizer.param_groups[0]['lr']

    print("Epoch %d: Train \nLearning Rate: %.6f" % (i, current_lr))
    train_loss, train_accuracy = run_1_epoch(dummy_model, loss_function1, loss_function2, optimizer, test_loader,
                                             train=True)

    # Update the learning rate scheduler
    scheduler.step()

    # Lists for train loss and accuracy for plotting
    train_loss_list.append(train_loss)
    train_accuracy_list.append(train_accuracy)

    # Validate the model on validation set
    print("Epoch %d: Validation" % (i))
    with torch.no_grad():
        val_loss, val_accuracy = run_1_epoch(dummy_model, loss_function1, loss_function2, optimizer, test_loader1,
                                             train=False)

    # Lists for val loss and accuracy for plotting
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_accuracy)
    wandb.log({"acc": val_accuracy, "loss": val_loss})
    print('train loss: %.4f' % (train_loss))
    print('val loss: %.4f' % (val_loss))
    print('train_accuracy %.2f' % (train_accuracy))
    print('val_accuracy %.2f' % (val_accuracy))

    # Save model if validation accuracy for current epoch is greater than
    # all the previous epochs
    if val_accuracy > val_accuracy_max:
        val_accuracy_max = val_accuracy
        print("New Max val Accuracy Acheived %.2f. Saving model.\n\n" % (val_accuracy_max))
        torch.save(dummy_model, 'best_val_acc_model.pth')
    else:
        print("val accuracy did not increase from %.2f\n\n" % (val_accuracy_max))

wandb.finish()
plt.figure()
plt.plot(train_accuracy_list, label="train_accuracy")
plt.plot(val_accuracy_list, label="val_accuracy")
plt.legend()

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.title('Training and val Accuracy')
plt.savefig('Training and val Accuracy')

plt.figure()
plt.plot(train_loss_list, label="train_loss")
plt.plot(val_loss_list, label="val_loss")

plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and val Loss')
plt.savefig('Training and val Loss')
