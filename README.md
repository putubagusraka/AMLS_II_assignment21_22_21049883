# AMLSII_21-22_SN21049883

## Introduction

In this assignment, we want to detect early signs of diabetic retinopathy, as seen in the Kaggle APTOS 2019 Blindness Detection Challenge.

This project explores the use of hybrid convolutional networks such as DenseNet-121 and VGG-16 with SVM.

## Organization of Files
```bash
├───Project Assets
│   ├───ablation_DenseNet_Model
│   │   ├───assets
│   │   └───variables
│   ├───ablation_VGG_Model
│   │   ├───assets
│   │   └───variables
│   ├───DenseNet_Model
│   │   ├───assets
│   │   └───variables
│   └───VGG_Model
│       ├───assets
│       └───variables
├───training_images
│   └───image (contains 3000 training images)   
├───Ablation_DenseNet.ipynb
├───Ablation_VGG16.ipynb
├───DenseNet.ipynb
├───VGG16.ipynb
├───project_functions.py
├───ablation_functions.py
├───__init__.py
├───train.csv
├───README.md
```

## File Roles
#### Base Directory

1) There are 4 main ipynb files (split due to computational restraints). Each can be separated into the 3 main processes of the assignment.

    a) For the main model training and testing, we have `DenseNet.ipynb` and `VGG16.ipynb`.
    
    b) For ablation studies (which are simply just slightly adjusted versions of the main training models), we have `Ablation_DenseNet.ipynb` and `Ablation_VGG16.ipynb`. 

2) The .py files `project_functions.py` and  `ablation_functions.py` are packaged modules that are imported when running the `Ablation_*.ipynb`,
    `DenseNet.ipynb`, and  `VGG16.ipynb` scripts.

3) The file folders `training_images/` acontain the 3000 training images. The file `train.csv` comtains the pair of each image and the class.

4) There is `Project Assets/` containing the trained models (used for testing or if user does not intend to rerun the training scripts), those being `ablation_DenseNet_Model/`, `ablation_VGG_Model/`, `DenseNet_Model/`, and `VGG_Model/`.

## Code Execution

### Step-by-step

1) Clone/download git repository (https://github.com/putubagusraka/AMLS_II_assignment21_22_21049883.git)
2) Download asset file from GDrive (https://drive.google.com/drive/folders/1M2L8bzbvg2wQUxOPRVJXmC2GlylzxS_L?usp=sharing)
3) Download image data from Kaggle APTOS 2019 Challenge (https://www.kaggle.com/c/aptos2019-blindness-detection/data?select=train.csv)
4) Extract asset file as is into main directory, foldering already as needed. (should resemble the file directory tree stated above)

#### Full run: Training and testing (both main and ablation)
1) Run `Ablation_DenseNet.ipynb` and `Ablation_VGG16.ipynb` scripts.
2) Run `DenseNet.ipynb` and `VGG16.ipynb` scripts.

## Dependent Environment and Libraries
The whole project is developed in Python 3.8.8. Please note that using other Python versions may lead to unknown errors. Required libraries are shown below.
* joblib==1.1.0
* keras==2.6.0
* matplotlib==3.4.3
* numpy==1.21.3
* opencv_python==4.5.3.56
* pandas==1.3.4
* scikit_learn==1.0.1
* tensorflow==2.7.0
* tensorflow_gpu==2.6.0
