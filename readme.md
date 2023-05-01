# Computer Vision CS-867 (Assignment #2)

Facial Expression recognition, and computing Valence and Arousal

**Muhammad Talha Imran MSDS-2022**

## Abstract

Human facial recognition is of key significance as human machine interfaces improve and the necessity to understand
human behavioral cues and better predict the outcomes. The degree of the emotion expressed in terms of arousal or
valence also hold significance rather than a crude classification of the emotion. With the advent of deep learning
techniques and their ability to extract complex features from the underlying distribution of data and subsequent
advancements with improved architectures over the years, it is now possible to gauge human emotion and to interpret the
degree of arousal or valence. In this assignment the provided task was to classify the given human facial expressions
and to regress the two attributes that quantify the degree of emotions i.e. valence and arousal. To complete this task
the backbones from certain pre-trained architectures were used as feature extractors while the subsequent fully
connected layers were modified and new regression heads were added in parallel to interpret and extract the degree of
emotion from images.

## Introduction

Deep learning provides a powerful set of tools to extract features from high dimensional data i.e. images whether they
are still frames or successive frames of a video stream. through the use of the convolutional mechanism, features or
areas of interest can be highlighted / exaggerated and the resulting feature map/ feature rich representation can be
subsequently used to perform classification or regression tasks. The applications are not only limited to classification
or regression problems but can be applied to segmentation, 3D reconstruction etc.

It is a well understood phenomena that successive convolutional layers extract successively higher order features i.e.
the first few convolutional layers tend to extract lower order features i.e. edges corners curves etc. while subsequent
layers tend to extract higher order features such as complex facial features, complex geometrical features or shapes
that pretain to specific objects. For example, while learning on images of airplanes the lower order features may
consist of edges and corners while higher order features may pretain to engines, wheels, flaps, wings etc. Thus, at the
end of the convolutional network we and up with a feature rich representation of the original input image. These
features maybe used in fully connected networks to complete the above-mentioned tasks. The convolutional layers may be
considered as building blocks for any number of computer vision applications mentioned above by only replacing the
output head that is the type of output layers used.

Over the years many extensive advancements have been made in the field of image classification, primarily it stemmed
from the image net large scale visual recognition challenge where over the years deep learning architectures have shown
advancements in classification performance attributed to the work of researchers globally. The result of many years of
this development is a family of diverse and capable architectures providing a vast array of choices for designers
working on solving image classification, regression, segmentation task etc. The designer needs only to use a suitable
backbone for their application and use a task specific head or a multitude of heads to achieve their desired output. Pre
Trained weights from the image net large scale visual recognition challenge are also available in modern frameworks that
allow the end user to find tune their model on their data set does allowing better generalization performance as well as
faster training times saving compute and energy resources. This transfer learning leverages the face that natural
objects tend to contain a vast array of diverse features that may overlap with other applications. Modern deep learning
frameworks allow for model editing and flexibility of modularity that enables such architectures to be used for any
conceivable computer vision application.

**Following is an illustration of the above described steps and provides a high level overview of the task:**
![Assignment_2_illustations_High_Level_Overview.png](Accumulated_data%2FAssignment_2_illustations_High_Level_Overview.png)

## Dataset and challenges:

The training dataset folder contains more than 280k images + each image has three annotation files containing the
classification value i.e. what class the image belongs to and the value for arousal and valence. For this assignment we
have chosen to ignore the official landmark invitation file as we attempt to leverage the ability of deep learning
architectures to learn and detect features within the images. The dataset itself presents a number of challenges,
primarily it presents a large imbalance amongst the classes.

**Below Figure illustrates the class imbalance in the training dataset.**
![Training_set_distribution.png](Accumulated_data%2FTraining_set_distribution.png)

In order to try to counteract the effect of this class imbalance during training, we employ a weighted cross entropy
loss function for classification, the weights of which are presented in figure below. The second issue posed by the data
set
are the annotations. The most major contributing factor to which is that human emotion maybe interpreted subjectively
from person to person i.e. fore example the annotator might interpret and emotion as contempt while others might
perceive it as neutral this is one example, other similar examples are also possible. Same is the case for valence and
arousal values that the degree of the expression for arousal maybe interpreted differently from person to person does
the network may have difficulty learning and some of our finding corroborate this assumption i.e. while learning we saw
the classification loss jump drastically while employing a relatively large batch size and relatively low learning rate
while achieving relatively low performance. There may be other factors at play as well, which will need additional
analysis to uncover. Bu the dataset's authors also present similar classification performance which leads us to this
assumption.

**Computed Weights for loss function based on the class imbalance using the sklearn library**
![Training_set_weight_distribution.png](Accumulated_data%2FTraining_set_weight_distribution.png)

Another challenge faced during training was of electrical grid instability where spontaneous loss of power presented a
hurdle in continous training, where in the training process was interuppted multiple times. Secondly the training was
done on a relatively old GPU with only 6 GB of video memory thus limiting the batch size that we could use. Internet
instability and large running time ruled out the feasibility of running on Google colab and kaggle notebooks. These
limitations resulted us in choosing comparatively smaller architectures with fewer number of parameters and ultimately
running fewer epoch. The results may improve have longer run times and trying deeper architectures with larger batch
sizes.

## Models implemented:

**Please note that more detailed discussion can be found in
the [Assignment Report](https://drive.google.com/drive/folders/1qWtsccwn86x9icn-79FX_FbDhpsNxEEn?usp=share_link)**

## Instructions for setting up the environment for running this implementation

### Installing dependencies:

```bash
  pip install -r requirements.txt
```

#### For Manual Installation

- albumentations==1.3.0
- krippendorff==0.6.0
- matplotlib==3.5.1
- numpy==1.21.5
- opencv_python_headless==4.7.0.72
- pandas==1.4.2
- Pillow==9.5.0
- scikit_learn==1.2.2
- torch==1.13.1+cu116
- torchmetrics==0.11.4
- torchsummary==1.5.1
- torchvision==0.14.1+cu116
- torchviz==0.0.2
- tqdm==4.64.0

## Instructions on running the train and test scripts on the train and test data

### Link to Dataset

[Download Dataset](https://drive.google.com/drive/folders/1BPioVa6zLIm8VbX_dftWQIC4fTe4veUY?usp=share_link)

**Create** a **Dataset** folder in the project and place the dataset in the dataset folder(ignored while uploading to
GitHub),
extract and place the contents in the dataset folder as shown below.

![Dataset_placement.PNG](Accumulated_data%2FDataset_placement.PNG)

### Dataset split used during training(80-20)

| Dataset    | Samples |
|------------|---------|
| Train      | 230120  |
| Test       | 57531   |
| Validation | 3999    |

## Running the implementation

##### Repository Directory Navigation

- [Accumulated_data](Accumulated_data) >> Contains illustrations for reporting and GitHub
- Dataset >> Contains the dataset
- [Models](Models) >> Models are saved to and loaded from this directory
- [Results](Results) >> Evaluation metrics and experiment outputs are saved here
- [SCR](SCR) >> contain the .py files for execution

![Directories.PNG](Accumulated_data%2FDirectories.PNG)

##### .py file navigation:

- CustomMobileNetV3.py >> **Class/model declaration** for **MobileNet** based model
- CustomModel.py >> **Class/model declaration** for **Custom Proposal** for model based on our assumptions
- CustomSqueezeNetModel.py >> **Class/model declaration** for **SqueezeNet** based model
- CustomEfficientNetModel.py >> **Class/model declaration** for **EfficientNet** based model
- FaceDataSet.py >> **Class declaration** for custom dataset
- HelperFunctions.py >> **Function declaration** for running epochs, running inference and running evaluation metrics
  etc.
- MobileNetV3.py >> **Model Execution** script for **MobileNet** based model
- SE_Net.py >> **Model Execution** script for **SqueezeNet** based model
- CustomNet.py >> **Model Execution** script for **Custom Proposal** model
- EfficientNet.py >> **Model Execution** script for **EfficientNet** based model
- test.py >> **upload ignore**, this was a testing file for functionality

### Control panel

Each script provides **4 major control elements** on what actions to preform:

![Program_Controls.PNG](Accumulated_data%2FProgram_Controls.PNG)

- **Run_training** >> **Boolean**, to execute training on the provided dataset, must be true even if resuming training,
  if
  you wish to get inference from trained model then set to False and the program will skip training
- **Resume_training** >> **Boolean**, to resume interrupted training set to True, else False. False will result in the
  model training from scratch if run_training is set to True. Presently, the auto resume functionality is not
  implemented and the **user will have to enter the starting epoch manually**(i.e. start from the ith epoch): Example
  use case is presented below. **Warning: To start from scratch and run the complete desired number of epochs, set
  starting_epoc to zero**
- **load_model** >> **Boolean**, to load previously saved model from the Models directory.
- **reu_test_set** >> **Boolean**, to run inference on the provided testset and generate performance metrics, saved to
  the Results directory

![Starting_epoch.PNG](Accumulated_data%2FStarting_epoch.PNG)

### Training Parameters

Training parameters can easily be set through the below-mentioned section:

![Training_parameters.PNG](Accumulated_data%2FTraining_parameters.PNG)

- **model_name** >> Name of the .py file running to standardize the names of the saved files and ease of later use
- **batch_size** >> sets the batch size param for dataloader
- **learning_rate** >> sets the starting learning rate for cosine annealing
- **pretrained** >> **Boolean**, This option is not valid for Custom model(proposed model),who's no pretrained weights
  exist on the ImageNet Dataset
- **epochs** >> Total Epochs to run, in case of resumed training, this is the ending epoch
- **classes** >> Used where number of classes needs to be referenced
- **device_name** >> torch.cuda.get_device_name(0) >> Name of the GPU used
- **device** >> Training device used
- **starting_time**>> logs the start time for duration calculation
- **class_names** >> used for reference while generating outputs

### Augmentations

![Augmentations.PNG](Accumulated_data%2FAugmentations.PNG)

- **mean** >> Per channel mean for normalization
- **std** >> Per channel standard deviation for normalization
- **augmentations** >> from the albumentations package

### Cross Entropy Weight Calculations

Calculation of the weights listed above, The calculation has been commented and the values for the dataset have been
hardcoded to save time. If any changes to the dataset are made, this should be uncommented for new weight calculation

![CrossEntropy_Weights.PNG](Accumulated_data%2FCrossEntropy_Weights.PNG)

### Loss Functions used

- **Classification** >> Cross Entropy Loss function
- **Regression** >> Mean Squared Error

Losses are added together before backpropagation in the below-mentioned ratio
**Loss = (90% Classification Loss) + (5% Valence Loss) + (5% Arousal Loss)**

These ratios were **found to work empirically** and hence used. They may be changed in the implementation as shown below

![Change_loss_ratios.png](Accumulated_data%2FChange_loss_ratios.png)

### Experiment Logging

Experiment logging us done using [Wandb](wandb.ai)

Wandb report are
available [here](https://drive.google.com/drive/folders/1O5k4GWzKOGFCsXAejOQUIcBfTclQXhxG?usp=share_link)

### Parameters of each Model are listed Below

Highlighted models are tested for this assignment due to the few number of parameters and the challenges listed above.

| Model             | Params      |
|-------------------|-------------|
| **Talha_Net**     | 423,498     |
| **SqueezeNet1.1** | 1,235,496   |
| **MobileNet**     | 2,118,954   |
| **Efficienet B0** | 4,020,358   |
| Googlenet         | 6,624,904   |
| RESNET18          | 11,689,512  |
| InceptionV3       | 27,161,264  |
| Efficienet V2M    | 54,139,356  |
| Efficienet V2     | 118,515,272 |
| VGG16             | 138,365,992 |
| ViT 14.00         | 633,470,440 |

### Model Visualization

TorchVis Visualizations of the models are available in the [Models directory](Models)

![Mobile_net_model.png](Models%2FMobile_net_model.png)

**T Net Architecture(Proposed)**

![Assignment_2_illustations_T_Net.png](Accumulated_data%2FAssignment_2_illustations_T_Net.png)


### Training

Best validation Accuracy model is saves i.e. stopping early.<br />
Examples: Train and Validation Accuracy for SE Net model:

![SE_Net Training and val Accuracy.png](Results%2FSE_Net%20Training%20and%20val%20Accuracy.png)
<br />

Examples Train and Validation Loss for SE Net model:

![SE_Net Training and val Loss.png](Results%2FSE_Net%20Training%20and%20val%20Loss.png)

Mentioned Below are the setting used for training

| Model                 | Initial Learning Rate | Final Learning Rate | LR Scheduler     | Batch Size | Epochs | Transfer Learning | Classifier Loss Function | Regression Loss Function | Execution Time per Epoch(avg mins) |
|-----------------------|-----------------------|---------------------|------------------|------------|--------|-------------------|--------------------------|--------------------------|------------------------------------|
| SE_Net                | 0.0001                | 0.0000001           | Cosine Annealing | 128        | 30     | FALSE             | Cross Entropy            | MSE                      | 51.8                               |
| T_Net(Proposed model) | 0.0001                | 0.0000001           | Cosine Annealing | 512        | 30     | NA                | Cross Entropy            | MSE                      | 43                                 |
| Eff_Net               | 0.0001                | 0.0000001           | Cosine Annealing | 32         | 12     | FALSE             | Cross Entropy            | MSE                      | 46                                 |
| Mobile_Net            | 0.0001                | 0.0000001           | Cosine Annealing | 512        | 12     | FALSE             | Cross Entropy            | MSE                      | 32                                 |

## Results

The [Results](Results) directory contains both quantitative and qualitative metrics run on each model.
**Nomenclature** is as follows:

- Classification_report_k_<**Model Name**>.csv >> SKLearn Classification report in .csv
- Confusion_matrix_<**Model Name**>.jpg >> Confusion Matrix
- <**Model Name**> Training and val Accuracy.png >> Training and Validation Accuracies over the training epochs
- <**Model Name**> Training and val Loss.png >> Training and Validation Losses over the training epochs
- <**Model Name**>_evaluation_metrics.csv >> Contains the saved evaluation metrics in .csv format
- <**Model Name**>_examples.png >> Set of 16 random test example images as qualitative measure
- <**Model Name**>_experiment_data.csv >> Contains the saved experimental data in .csv format

![Results_Dir.PNG](Accumulated_data%2FResults_Dir.PNG)

**Example Confusion Matrix for SE-NET**

![Confusion_matrix_SE_Net.jpg](Results%2FConfusion_matrix_SE_Net.jpg)