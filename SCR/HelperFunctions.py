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
import matplotlib.pyplot as plt
import pandas as pd
import glob
from sklearn.utils.class_weight import compute_class_weight as compute_weights
from collections import Counter


def run_epoch(model, classification_loss_function, regression_loss_function, optimizer, data_loader, train=False,
              device="cuda", loss_weights=[1, 1, 1], logger=None):
    """
    Runs one complete epoch using the provided model on the provided dataloader
    :param classification_loss_function:
    :param model: Torch model
    :param classification_loss_function: chosen classification loss function
    :param regression_loss_function: chosen regression loss function
    :param optimizer: chosen optimizer
    :param data_loader: dataloader
    :param train: if true, preforms training, else can be used for test accuracy
    :param device: GPU or CPU
    :param loss_weights: if the user wants to weight the losses in a specific ratio(python list of int/floats,in [classification, arousal, valence])
    :param logger: Wandb object to log loss while training. If you dont want to log losses, None
    :return: total loss, classification loss, arousal loss, valence loss, accuracy
    """
    if train:
        model.train()
    else:
        model.eval()

    correct_predictions = 0
    total_loss = 0
    dataset_batches = data_loader.dataset.__len__()
    batch_samples = len(data_loader)
    for image, label, arousal, valence in tqdm(data_loader):
        image = image.to(device)
        label = label.to(device)
        arousal = arousal.to(device)
        valence = valence.to(device)

        if train:
            optimizer.zero_grad()

        output_class, output_arousal, output_valence = model(image)

        _, predicted = torch.max(output_class.data, 1)  # the dimension 1 corresponds to max along the rows

        # to run SE-Net comment this, else remains uncommented
        # output_class.squeeze_(-1)
        arousal = arousal.unsqueeze(1)
        valence = valence.unsqueeze(1)

        classification_loss = loss_weights[0] * classification_loss_function(output_class, label)
        arousal_loss = loss_weights[1] * regression_loss_function(output_arousal, arousal)
        valence_loss = loss_weights[2] * regression_loss_function(output_valence, valence)

        loss = classification_loss + arousal_loss + valence_loss

        if train:
            loss.backward()

        if train:
            optimizer.step()

        correct_predictions += (predicted == label).sum().item()

        total_loss += loss.item()
        if logger is not None:
            logger.log({"cLassification_loss": classification_loss,
                        "arousal_loss": arousal_loss, "valence_loss": valence_loss})

    loss = total_loss / batch_samples
    accuracy = 100 * correct_predictions / dataset_batches

    return loss, classification_loss, arousal_loss, valence_loss, accuracy


def run_test(model, data_loader, device):
    """
    Run forward pass(inference) on the passed dataloader using the provided model
    :param model: Pytorch model(trained)
    :param data_loader: Dataloader to be tested upon
    :param device: "GPU"/"CPU"
    :return: Predicted(class,arousal,valence),Ground Truth(class,arousal,valence)
    """
    # model.to(device)
    predicted_classifications = []
    predicted_arousal = []
    predicted_valence = []
    groundtruth_classification = []
    groundtruth_arousal = []
    groundtruth_valence = []
    for image, label, arousal, valence in tqdm(data_loader):
        image = image.to(device)
        label = label.to(device)
        arousal_ = arousal.to(device)
        valence_ = valence.to(device)
        output_class, output_arousal, output_valence = model(image)

        _, predicted = torch.max(output_class.data, 1)  # the dimension 1 corresponds to max along the rows

        output_class.squeeze_(-1)
        arousal = output_arousal.squeeze(-1)
        valence = output_valence.squeeze(-1)
        predicted_classifications.extend(predicted.cpu().detach().numpy())
        predicted_arousal.extend(output_arousal.squeeze(-1).cpu().detach().numpy()[:])
        predicted_valence.extend(output_valence.squeeze(-1).cpu().detach().numpy()[:])
        groundtruth_classification.extend(label.cpu().numpy())
        groundtruth_arousal.extend(arousal_.cpu().numpy())
        groundtruth_valence.extend(valence_.cpu().numpy())

    return predicted_classifications, predicted_arousal, predicted_valence, groundtruth_classification, groundtruth_arousal, groundtruth_valence


def save_classification_report(report, path, model_name):
    """
    Save the classification report as csv
    :param report: Report to save
    :param path: Where to save the report
    :param model_name: Name of the model
    :return: Nothing
    """
    df = pd.DataFrame(report)
    df.to_csv(path + "Classification_report_" + "k_" + f"{model_name}.csv")
    return 0


def display_examples(images, actual_labs, predicted_labs):
    """
    To display examples of predictions on images
    :param images: Images to display
    :param actual_labs: Correct labels
    :param predicted_labs: Predicted labels from the classifier
    :return: Nothing
    """
    p = len(images)
    q = 2
    p = p // 2
    i = 1
    for img in images:
        plt.subplot(q, p, i)
        plt.imshow(img, cmap="gray")
        plt.title("Actual lab: " + actual_labs[i - 1] + ", "
                  + "Prediction: " + predicted_labs[i - 1])
        i += 1
    plt.show()
    return 0


def compute_class_weights(dataset_path, class_names):
    """
    Computes the per class weights for the loss function
    :param dataset_path: path to dataset annotation files
    :param class_names: Names of the classes
    :return: Weights for loss function
    """
    classes = np.zeros(8)  # 8 classes
    labels = []
    for file in glob.glob(dataset_path + "/" + "*_exp.npy"):
        lab = int(np.load(file))
        classes[lab] += 1
        labels.append(str(lab))

    # plotting the training set distribution
    xticks = range(8)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x=xticks, height=classes)
    ax.set_xticks(xticks, class_names)
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.margins(y=0.1)

    ax.set_xlabel('Class')
    ax.set_ylabel('Samples')
    ax.set_title('Training set distribution')
    plt.savefig("../Results/" + ' Training set distribution.png')
    # computing weights using the sklearn library
    weights = compute_weights(class_weight='balanced', classes=np.unique(labels),
                              y=pd.DataFrame(labels).values.reshape(-1))

    # plotting the training set distribution
    xticks = range(8)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x=xticks, height=weights)
    ax.set_xticks(xticks, class_names)
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.margins(y=0.1)
    # ax.legend()
    ax.set_xlabel('Class')
    ax.set_ylabel('Weights')
    ax.set_title('Training set weight distribution')
    plt.savefig("../Results/" + ' Training set weight distribution.png')

    return weights


def print_model_summery(model, input_size=[3, 224, 224]):
    """
    Prints out the summary for the model
    :param input_size: Size of the input used with model, default set to [3,224,224]
    :param model: Pytorch model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.cuda()
    x = torch.rand(size=input_size)
    summary(model, input_size)
