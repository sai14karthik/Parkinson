

# Parkinson Detection usinf DeepNets Repository 

## 📌 Overview

This repository contains a project aimed at detecting Parkinson's Disease using various deep learning models applied to MRI data. The project leverages state-of-the-art neural network architectures to analyze medical imaging data and predict the presence of Parkinson's Disease.

## 📁 Repository Structure

The repository is organized as follows:

- **`parkinsons_dataset/`**: Contains the dataset used for training and evaluation.
- **`.DS_Store`**: System file (can be ignored).
- **`parkinson-mri_desnet121.ipynb`**: Jupyter Notebook implementing the DenseNet121 model.
- **`parkinson-mri_efficientnetb0.ipynb`**: Implementation of EfficientNetB0.
- **`parkinson-mri_efficientnetb7.ipynb`**: Implementation of EfficientNetB7.
- **`parkinson-mri_inceptionv3.ipynb`**: Implementation of InceptionV3.
- **`parkinson-mri_mobilenet.ipynb`**: Implementation of MobileNet.
- **`parkinson-mri_nasnetmobile.ipynb`**: Implementation of NASNetMobile.
- **`parkinson-mri_resnet50.ipynb`**: Implementation of ResNet50.
- **`parkinson-mri_vgg16.ipynb`**: Implementation of VGG16.
- **`parkinson-mri_vgg19.ipynb`**: Implementation of VGG19.
- **`parkinson-mri_xception.ipynb`**: Implementation of Xception.


## 🚀 Features

1. Utilizes multiple deep learning models for Parkinson's Disease detection:
    - DenseNet121
    - EfficientNet (B0, B7)
    - InceptionV3
    - MobileNet
    - NASNetMobile
    - ResNet50
    - VGG (16, 19)
    - Xception
2. Focuses on MRI data analysis for medical imaging applications.
3. Implements Jupyter Notebooks for easy experimentation and visualization.

## 🛠️ Setup and Installation

To run the project, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/sai14karthik/Parkinson.git
cd Parkinson
```

2. Install the required dependencies:
    - Python 3.x is required.
    - Install dependencies using pip or conda:

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn jupyter
```

3. Open Jupyter Notebook:

```bash
jupyter notebook
```

Navigate to the desired `.ipynb` file and run the cells.

## 📊 Dataset

The dataset used in this project is stored in the `parkinsons_dataset/` directory. Ensure that the dataset is preprocessed and formatted correctly before running the notebooks.

**[Parkinson's Brain MRI Dataset](https://www.kaggle.com/datasets/irfansheriff/parkinsons-brain-mri-dataset)**

## ⚙️ Usage

1. Choose a model notebook (e.g., `parkinson-mri_resnet50.ipynb`) and open it in Jupyter Notebook.
2. Follow the steps in the notebook to load the dataset, preprocess it, train the model, and evaluate its performance.
3. Modify hyperparameters or experiment with different architectures as needed.

## 📈 Results

Each notebook provides detailed metrics such as accuracy, precision, recall, and F1-score for evaluating model performance on detecting Parkinson's Disease.

