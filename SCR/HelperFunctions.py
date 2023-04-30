import torch
from torch import nn
import torchvision
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
from torchsummary import summary
from FaceDataSet import FaceDataset
from torch.utils.data import DataLoader
import cv2
from torchvision.transforms import transforms as torch_transformations
import matplotlib.pyplot as plt
import pandas as pd
import glob
from sklearn.utils.class_weight import compute_class_weight as compute_weights
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import average_precision_score, mean_squared_error, auc
import krippendorff
from collections import Counter


def run_epoch(model, classification_loss_function, regression_loss_function, optimizer, data_loader, train=False,
              device="cuda", loss_weights=[1, 1, 1], logger=None):
    """
    Runs one complete epoch using the provided model on the provided dataloader
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


def display_examples(dataloader, model_name, class_names, predicted_classifications, predicted_arousal,
                     predicted_valence,
                     groundtruth_classification, groundtruth_arousal, groundtruth_valence):
    """
    Displays example images, along with predicted and ground truth labels and regression values
    :param dataloader: Examples dataloader
    :param model_name: Torch model to test
    :param class_names: Names of classes under consideration[list]
    :param predicted_classifications:Network output classes[list]
    :param predicted_arousal: Network output arousal values[list]
    :param predicted_valence: Network output valence values[list]
    :param groundtruth_classification:Groundtruth labels[list]
    :param groundtruth_arousal: Groundtruth arousal values[list]
    :param groundtruth_valence: Groundtruth valence values[list]
    :return: Nothing
    """
    images = []
    for image_batch, _, _, _ in dataloader:
        # Preforming un-normalization to restore color
        reverse_norm = torch_transformations.Compose([torch_transformations.Normalize(mean=[0., 0., 0.],
                                                                                      std=[1 / 0.229, 1 / 0.224,
                                                                                           1 / 0.225]),
                                                      torch_transformations.Normalize(mean=[-0.485, -0.456, -0.406],
                                                                                      std=[1., 1., 1.]),
                                                      ])
        image_batch = reverse_norm(image_batch)
        image_batch = image_batch.numpy()
        for image in range(np.shape(image_batch)[0]):
            images.append(image_batch[image, :, :, :])
    p = len(images)
    q = 4
    p = 4
    i = 1
    plt.figure(figsize=(10, 10))
    plt.title("Examples: " + model_name)
    for img in images:
        plt.subplot(q, p, i)
        plt.imshow((cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_BGR2RGB) * 255))
        plt.title("Actual lab: " + class_names[groundtruth_classification[i - 1]] + ", "
                  + "Prediction: " +
                  class_names[groundtruth_classification[i - 1]] + "\n"
                  + "Actual Arousal: " +
                  str(round(groundtruth_arousal[i - 1], 4)) + ", "
                  + "Predicted Arousal: " +
                  str(round(predicted_arousal[i - 1], 4)) + "\n"
                  + "Actual valence: " +
                  str(round(groundtruth_valence[i - 1], 4)) + ", "
                  + "Predicted valence: " +
                  str(round(predicted_valence[i - 1], 4)))
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
    # counting the labels of each class
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


# Reference for this metric https://github.com/stylianos-kampakis/supervisedPCA-Python/blob/master/Untitled.py
def concordance_correlation_coefficient(y_true, y_pred,
                                        sample_weight=None,
                                        multioutput='uniform_average'):
    """Concordance correlation coefficient.
    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    --------
    # >>> from sklearn.metrics import concordance_correlation_coefficient
    # >>> y_true = [3, -0.5, 2, 7]
    # >>> y_pred = [2.5, 0.0, 2, 8]
    # >>> concordance_correlation_coefficient(y_true, y_pred)
    0.97678916827853024
    """
    cor = np.corrcoef(y_true, y_pred)[0][1]
    y_pred = np.asarray(y_pred)
    y_true = np.array(y_true)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator


def sign_agreement(y_true, y_pred):
    """
    Calculates the sign agreement metric between form true and predicted labels.
    :param y_true: True labels[list]
    :param y_pred: Predicted lables[list]
    :return: Sign agreement metric.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.array(y_true)
    s = np.sign((y_true) - (y_pred))
    signed_diff = s * (y_true - y_pred)
    return np.mean(signed_diff)


def save_preformance_metric(model_name, class_names, predicted_classifications, predicted_arousal, predicted_valence,
                            groundtruth_classification, groundtruth_arousal, groundtruth_valence):
    """
    Saves and displays Preformance mertics
    :param model_name: Pytorch model to test
    :param class_names: Names of classes under consideration[list]
    :param predicted_classifications:Network output classes[list]
    :param predicted_arousal: Network output arousal values[list]
    :param predicted_valence: Network output valence values[list]
    :param groundtruth_classification:Groundtruth labels[list]
    :param groundtruth_arousal: Groundtruth arousal values[list]
    :param groundtruth_valence: Groundtruth valence values[list]
    :return: Nothing
    """

    # Confusion Matrix
    confusion_matrix_ = confusion_matrix(groundtruth_classification, predicted_classifications)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_, display_labels=class_names)
    display.plot()
    plt.title(f"Confusion_matrix_{model_name}")
    plt.savefig(f"../Results/Confusion_matrix_{model_name}.jpg")

    # used from this reference https://github.com/pln-fing-udelar/fast-krippendorff
    krippendorff_data = np.array([groundtruth_classification, predicted_classifications])
    krippendorff_aplha = krippendorff.alpha(reliability_data=krippendorff_data,
                                            level_of_measurement="nominal")

    # compute Cohens Kappa score
    cohens_kappa = cohen_kappa_score(groundtruth_classification, predicted_classifications)
    # print("AUC: ",
    #       auc(groundtruth_classification, predicted_classifications))
    # print("AUC – Precision Recall: ",
    #       average_precision_score(groundtruth_classification, predicted_classifications))

    # Reusing the function from Assignment_1, savinf the classificiation report
    report = classification_report(groundtruth_classification, predicted_classifications,
                                   target_names=class_names, output_dict=True)
    save_classification_report(report=report, path="../Results/", model_name=model_name)

    # Root mean squared error
    MSE_Arousal = mean_squared_error(groundtruth_arousal, predicted_arousal, squared=False)
    MSE_Valence = mean_squared_error(groundtruth_valence, predicted_valence, squared=False)
    # corralation correlation coefficients
    Correlation_Arousal = np.corrcoef(groundtruth_arousal, predicted_arousal)
    Correlation_Valence = np.corrcoef(groundtruth_valence, predicted_valence)
    # Sign Agreement Metric
    SAGR_Arousal = sign_agreement(groundtruth_arousal, predicted_arousal)
    SAGR_Valence = sign_agreement(groundtruth_valence, predicted_valence)
    # Concordance Correlation Coefficient
    CCC_Arousal = concordance_correlation_coefficient(groundtruth_arousal, predicted_arousal)
    CCC_Valence = concordance_correlation_coefficient(groundtruth_valence, predicted_valence)
    # Compiling it all into a dataframe
    output_report_data = [
        {"Model/Backbone": model_name, "krippendorff's aplha": krippendorff_aplha, "cohen's kappa": cohens_kappa,
         "RMSE_Arousal": MSE_Arousal, "RMSE_Valence": MSE_Valence, "Correlation_Arousal": Correlation_Arousal[0, 1],
         "Correlation_Valence": Correlation_Valence[0, 1], "SAGR_Arousal": SAGR_Arousal, "SAGR_Valence": SAGR_Valence,
         "CCC_Arousal": CCC_Arousal, "CCC_Valence": CCC_Valence}]

    output_dataframe = pd.DataFrame(output_report_data)

    # Saving the dataframe in the results folder
    output_dataframe.to_csv("../Results/" + model_name + "_evaluation_metrics.csv")
