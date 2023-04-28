import torch
from torch import nn
import torchvision
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
from torchvision import models
from torchsummary import summary
from torchmetrics.classification import MulticlassCohenKappa, MulticlassAUROC, MulticlassF1Score, MulticlassRecall, \
    MulticlassPrecision, MulticlassConfusionMatrix, MulticlassROC
from FaceDataloader import FaceDataset
from torch.utils.data import DataLoader
import cv2
import albumentations as Augmentations
import os
import matplotlib.pyplot as plt
import wandb
import pandas as pd
from CustomVGGModel import VGG16_model
from HelperFunctions import run_epoch, run_test, save_classification_report, display_examples
import time
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import average_precision_score, mean_squared_error, auc
import krippendorff

# These params can be updated to allow for better control of the program(i.e. the control knobs of this code)
run_training = True  # False to run inferences, otherwise it'll start train the model
resume_training = False  # If training needs to be resumed from some epoch
load_model = False  # If you want to load a model previously trained
run_test_set = False  # True to run test set post training

model_name = os.path.basename(__file__).split(".")[
    0]  # Name of the .py file running to standardize the names of the saved files and ease of later use
batch_size = 8
learning_rate = 0.001
pretrained = True
epochs = 10
classes = 8
device_name = torch.cuda.get_device_name(0)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
starting_time = time.time()
class_names = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

# Path to the dataset
train_image_path = "../Dataset/train_set/images"
train_annotation_path = "../Dataset/train_set/annotations"
validation_image_path = "../Dataset/val_set/images"
validation_annotation_path = "../Dataset/val_set/annotations"

# Paths to store generated data
model_save_path = "../Models"
data_save_path = "../Accumulated_data"
figure_save_path = "../Results"

# These values come from imagenet, i.e. leverage the relationships of natural objects
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

augmentations = Augmentations.Compose([Augmentations.HorizontalFlip(), Augmentations.VerticalFlip(),
                                       Augmentations.GridDistortion(p=0.2),
                                       Augmentations.GaussNoise()])

train_set = FaceDataset(train_image_path, train_annotation_path, mean, std, augmentations)
train_size = int(0.8 * len(train_set))
test_size = len(train_set) - train_size
train_set, test_set = torch.utils.data.random_split(train_set, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

validation_set = FaceDataset(validation_image_path, validation_annotation_path, mean, std, augmentations)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

# only for testing, this loop is meant to loop error verification before committing to full length training
# mini_size = int(0.5 * len(validation_set))
# mini_test_size = len(validation_set) - mini_size
# mini_dataset, mini_test_set = torch.utils.data.random_split(validation_set, [mini_size, mini_test_size])
# mini_loader = DataLoader(mini_dataset, batch_size=batch_size, shuffle=True)
# mini_test_loader = DataLoader(mini_test_set, batch_size=batch_size, shuffle=True)

vgg_model = VGG16_model(pretrained=pretrained).to(device)

regression_loss_function = torch.nn.MSELoss()
classification_loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg_model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)
if run_training:
    train_accuracy_list = []
    val_accuracy_list = []
    train_classification_loss_list = []
    train_arousal_loss_list = []
    train_valence_loss_list = []

    train_loss_list = []
    val_loss_list = []
    val_classification_loss_list = []
    val_arousal_loss_list = []
    val_valence_loss_list = []
    starting_epoch = 0
    val_accuracy_max = -1

    wandb.init(
        # set the wandb project where this run will be logged
        project="Custom_model-test",

        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "VGG_16_test",
            "dataset": "Faces",
            "epochs": epochs,
        }
    )
    if resume_training:
        checkpoint = torch.load("../Models/" + model_name + '_best_val_acc_model.pth')
        vgg_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # starting_epoch.load_state_dict(checkpoint["epoch"])
        # train_loss_list.load_state_dict(checkpoint['train_losses'])
        # train_accuracy_list.load_state_dict(checkpoint['train_accuracies'])
        # val_loss_list.load_state_dict(checkpoint['val_losses'])
        # val_accuracy_list.load_state_dict(checkpoint['val_accuracies'])
        # val_classification_loss_list.load_state_dict(checkpoint['classification_loss'])
        # val_arousal_loss_list.load_state_dict(checkpoint["arousal loss"])
        # val_valence_loss_list.load_state_dict(checkpoint['valence loss'])
    for epoch in range(starting_epoch, epochs):

        # Train model for one epoch

        train_loss, classification_loss, arousal_loss, valence_loss, train_accuracy = run_epoch(vgg_model,
                                                                                                classification_loss_function,
                                                                                                regression_loss_function,
                                                                                                optimizer,
                                                                                                train_loader,
                                                                                                train=True,
                                                                                                device=device,
                                                                                                loss_weights=[0.9, 0.05,
                                                                                                              0.05],
                                                                                                logger=wandb)

        # Update the learning rate scheduler
        scheduler.step()

        # Lists for train loss and accuracy for plotting
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        train_arousal_loss_list.append(arousal_loss)
        train_classification_loss_list.append(classification_loss)
        train_valence_loss_list.append(valence_loss)

        # Validate the model on validation set
        print("Epoch %d: Validation" % (epoch))
        with torch.no_grad():
            val_loss, classification_loss, arousal_loss, valence_loss, val_accuracy = run_epoch(vgg_model,
                                                                                                classification_loss_function,
                                                                                                regression_loss_function,
                                                                                                optimizer,
                                                                                                validation_loader,
                                                                                                train=False,
                                                                                                device=device,
                                                                                                loss_weights=[0.9, 0.05,
                                                                                                              0.05],
                                                                                                logger=None)

        # Lists for val loss and accuracy for plotting
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)
        val_classification_loss_list.append(classification_loss)
        val_valence_loss_list.append(valence_loss)
        val_arousal_loss_list.append(arousal_loss)
        wandb.log({"acc": val_accuracy, "training_loss": train_loss, "validation_loss": val_loss})
        print('train loss: %.4f' % (train_loss))
        print('val loss: %.4f' % (val_loss))
        print('train_accuracy %.2f' % (train_accuracy))
        print('val_accuracy %.2f' % (val_accuracy))

        # Save model if validation accuracy for current epoch is greater than
        # all the previous epochs
        if val_accuracy > val_accuracy_max:
            val_accuracy_max = val_accuracy
            print("New Max val Accuracy Acheived %.2f. Saving model.\n\n" % (val_accuracy_max))
            checkpoint = {
                'model': vgg_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'trianed_epochs': epoch,
                'train_losses': train_loss_list,
                'train_accuracies': train_accuracy_list,
                'val_losses': val_loss_list,
                'val_accuracies': val_accuracy_list,
                'classification_loss': val_classification_loss_list,
                "arousal loss": val_arousal_loss_list,
                'valence loss': val_valence_loss_list,
                'lr': learning_rate
            }
            torch.save(checkpoint, "../Models/" + model_name + '_best_val_acc_model.pth')

        else:
            checkpoint = {
                'model': vgg_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'trianed_epochs': epoch,
                'train_losses': train_loss_list,
                'train_accuracies': train_accuracy_list,
                'val_losses': val_loss_list,
                'val_accuracies': val_accuracy_list,
                'classification_loss': val_classification_loss_list,
                "arousal loss": val_arousal_loss_list,
                'valence loss': val_valence_loss_list,
                'lr': learning_rate
            }
            torch.save(checkpoint, "../Models/" + model_name + '_best_val_acc_model.pth')
            print("val accuracy did not increase from %.2f\n\n" % (val_accuracy_max))

    wandb.finish()
    execution_time = time.time() - starting_time
    plt.figure()
    plt.plot(train_accuracy_list, label="train_accuracy")
    plt.plot(val_accuracy_list, label="val_accuracy")
    plt.legend()

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.title(model_name + ' Training and val Accuracy')
    plt.savefig("../Results/" + model_name + ' Training and val Accuracy.png')

    plt.figure()
    plt.plot(train_loss_list, label="train_loss")
    plt.plot(val_loss_list, label="val_loss")

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(model_name + ' Training and val Loss')
    plt.savefig("../Results/" + model_name + " Training and val Loss.png")

    output_report_data = [{"Model/Backbone": model_name, "Learning_rate": learning_rate, "Batch_size": batch_size,
                           "Training_set_size": train_loader.dataset.__len__(), "Epochs": epochs,
                           "Test_set_size": test_loader.dataset.__len__(),
                           "Validation_set_size": validation_loader.dataset.__len__(), "Tranfer_learning": pretrained,
                           "Max Validation_Acc": val_accuracy_max,
                           "Classification_loss": val_classification_loss_list[-1],
                           "Arousal_loss": val_arousal_loss_list[-1], "Valence_loss": val_valence_loss_list[-1],
                           "Execution_time": execution_time}]

    output_dataframe = pd.DataFrame(output_report_data)
    output_dataframe.to_csv("../Results/" + model_name + "_experiment_data.csv")

# Running predictions on the test_set

if run_test_set:
    if load_model:
        # 'model': vgg_model.state_dict(),
        # 'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict(),
        # 'trianed_epochs': epoch,
        # 'train_losses': train_loss_list,
        # 'train_accuracies': train_accuracy_list,
        # 'val_losses': val_loss_list,
        # 'val_accuracies': val_accuracy_list,
        # 'classification_loss': val_classification_loss_list,
        # "arousal loss": val_arousal_loss_list,
        # 'valence loss': val_valence_loss_list,
        # 'lr': learning_rate
        checkpoint = torch.load("../Models/" + model_name + '_best_val_acc_model.pth')
        vgg_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # epoch.load_state_dict()
        # epoch
        # train_loss_list
        # train_accuracy_list
        # val_loss_list
        # val_accuracy_list
        # val_classification_loss_list
        # val_arousal_loss_list
        # val_valence_loss_list
        # learning_rate

        vgg_model.eval()

    vgg_model.to(device)
    predicted_classifications, predicted_arousal, predicted_valence, groundtruth_classification, groundtruth_arousal, groundtruth_valence = run_test(
        vgg_model, validation_loader, device)

    # confusion_matrix = MulticlassConfusionMatrix(num_classes=classes)
    # confusion_matrix = confusion_matrix(groundtruth_classification, predicted_classifications)
    print(groundtruth_classification, predicted_classifications)
    confusion_matrix = confusion_matrix(groundtruth_classification, predicted_classifications)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=class_names)
    display.plot()
    plt.title(f"Confusion_matrix_{model_name}")
    plt.savefig(f"../Results/Confusion_matrix_{model_name}.jpg")
    krippendorff_data = np.array([groundtruth_classification, predicted_classifications])
    print("Krippendorff's alpha for nominal metric: ", krippendorff.alpha(reliability_data=krippendorff_data,
                                                                          level_of_measurement="nominal"))
    print("cohen_kappa_score: ", cohen_kappa_score(groundtruth_classification, predicted_classifications))
    print("AUC: ",
          auc(groundtruth_classification, predicted_classifications))
    print("AUC – Precision Recall: ",
          average_precision_score(groundtruth_classification, predicted_classifications))

    # Reusing the function from Assignment_1
    report = classification_report(groundtruth_classification, predicted_classifications,
                                   target_names=class_names, output_dict=True)
    save_classification_report(report=report, path="../Results/", model_name=model_name)

    print("MSE Arousal: ", mean_squared_error(groundtruth_arousal, predicted_arousal))
    print("MSE Valence: ", mean_squared_error(groundtruth_valence, predicted_valence))

    print("Correlation Arousal: ", np.corrcoef((groundtruth_arousal), (predicted_arousal)))
    print("Correlation Valence: ", np.corrcoef((groundtruth_valence), (predicted_valence)))
