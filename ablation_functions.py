import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import keras

from collections import Counter

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,precision_score,f1_score,recall_score
from sklearn import preprocessing
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.applications import DenseNet121

import matplotlib.pyplot as plt

import pickle
import joblib
from joblib import dump, load

def ablation_build_VGG(image_size):
    """
    Build the VGG-16 and SVM Model.
    Callable for the the training process.
    
    """
    model = Sequential()
    model.add(Conv2D(input_shape=(image_size,image_size,3),filters=32,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(units=4096,activation="relu"))
    model.add(BatchNormalization())
    
    model.add(Dense(units=4096,activation="relu"))
    model.add(BatchNormalization())
    
    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(3, activation='softmax'))
    model.compile(
        loss='categorical_hinge',
        optimizer=Adam(learning_rate=0.001),
        metrics=['acc']
    )
    
    return model

def ablation_build_densenet(image_size):
    
    """
    Build the DenseNet-121 and SVM Model.
    Callable for the the training process.
    
    """
    densenet = DenseNet121(
    weights='Project Assets/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(image_size,image_size,3)
    )
    model = Sequential()
    model.add(densenet)
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    
    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(3, activation='softmax'))
    model.compile(
        loss='categorical_hinge',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['acc']
    )
    
    return model

def ablation_train_VGG(x_train_val, y_train_val, image_size):
    """
    This is the VGG training function with 3-fold cross validation.
    Here accuracy, precision, recall, and f1 score are tallied and average for the 3 models
    The CNN model is saved to local directory for any further testing
    
    Key Parameters:
    x_train_val = the training set as result from the preprocessing function, split into training and validation data 
    y_train_val = the supervised labels for training set as result from the preprocessing function, split into training and validation data 
    image_size = image resolution
    
    """
    ohe = joblib.load('Project Assets/ohe_encoder.joblib')

    kf = KFold(n_splits=3,shuffle=True)
    k_number = 0
    
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_f1score = []
    
    
    print("VGG training with 3-Fold Cross Validation.")
    for train_index, test_index in kf.split(x_train_val):
        x_train, x_val = x_train_val[train_index], x_train_val[test_index]
        y_train, y_val = y_train_val[train_index], y_train_val[test_index]
        
        model = ablation_build_VGG(image_size)
        
        history1 = model.fit(x_train,y_train,epochs=15, validation_split = 0.1)
        
        acc_history = history1.history['acc']
        val_acc_history = history1.history['val_acc']
        loss_history = history1.history['loss']
        val_loss_history = history1.history['val_loss']
        
        plt.plot(history1.history['acc'])
        plt.plot(history1.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history1.history['loss'])
        plt.plot(history1.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        

        print("The highest validation acc is {}".format(np.max(val_acc_history)))
        
        result=model.predict(x_val)
        result_class = tf.one_hot(np.argmax(result, axis=1), depth = 3)
        
        result_class = ohe.inverse_transform(result_class)
        y_val_class = ohe.inverse_transform(y_val)

        val_accuracy.append(accuracy_score(result_class, y_val_class))
        val_precision.append(precision_score(result_class, y_val_class,average='macro'))
        val_f1score.append(f1_score(result_class, y_val_class,average='macro'))
        val_recall.append(recall_score(result_class, y_val_class,average='macro'))

        average_val_accuracy=sum(val_accuracy)/len(val_accuracy)
        average_val_precision=sum(val_precision)/len(val_precision)
        average_val_recall=sum(val_recall)/len(val_recall)
        average_val_f1score=sum(val_f1score)/len(val_f1score)
        
        print("VGG 3-Fold CV:")
        print("Average Acc: %.4f" %(average_val_accuracy))
        print("Average Precision: %.4f" %(average_val_precision))
        print("Average recall: %.4f" %(average_val_recall))
        print("Average F1 Score: %.4f \n" %(average_val_f1score))
        
    model.save('Project Assets/ablation_VGG_Model')
    
def ablation_train_densenet(x_train_val, y_train_val, image_size):
    """
    This is the DenseNet training function with 3-fold cross validation.
    Here accuracy, precision, recall, and f1 score are tallied and average for the 3 models
    The CNN model is saved to local directory for any further testing
    
    Key Parameters:
    x_train_val = the training set as result from the preprocessing function, split into training and validation data 
    y_train_val = the supervised labels for training set as result from the preprocessing function, split into training and validation data 
    image_size = image resolution
    
    """
    ohe = joblib.load('Project Assets/ohe_encoder.joblib')

    kf = KFold(n_splits=3,shuffle=True)
    k_number = 0
    
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_f1score = []
    
    
    print("DenseNet training with 3-Fold Cross Validation.")
    for train_index, test_index in kf.split(x_train_val):
        x_train, x_val = x_train_val[train_index], x_train_val[test_index]
        y_train, y_val = y_train_val[train_index], y_train_val[test_index]
        
        model = ablation_build_densenet(image_size)
        
        history1 = model.fit(x_train,y_train,epochs=15, validation_split = 0.1)
        
        acc_history = history1.history['acc']
        val_acc_history = history1.history['val_acc']
        loss_history = history1.history['loss']
        val_loss_history = history1.history['val_loss']
        
        plt.plot(history1.history['acc'])
        plt.plot(history1.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history1.history['loss'])
        plt.plot(history1.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        

        print("The highest validation acc is {}".format(np.max(val_acc_history)))
        
        result=model.predict(x_val)
        result_class = tf.one_hot(np.argmax(result, axis=1), depth = 3)
        
        result_class = ohe.inverse_transform(result_class)
        y_val_class = ohe.inverse_transform(y_val)

        val_accuracy.append(accuracy_score(result_class, y_val_class))
        val_precision.append(precision_score(result_class, y_val_class,average='macro'))
        val_f1score.append(f1_score(result_class, y_val_class,average='macro'))
        val_recall.append(recall_score(result_class, y_val_class,average='macro'))

        average_val_accuracy=sum(val_accuracy)/len(val_accuracy)
        average_val_precision=sum(val_precision)/len(val_precision)
        average_val_recall=sum(val_recall)/len(val_recall)
        average_val_f1score=sum(val_f1score)/len(val_f1score)
        
        print("DenseNet 3-Fold CV:")
        print("Average Acc: %.4f" %(average_val_accuracy))
        print("Average Precision: %.4f" %(average_val_precision))
        print("Average recall: %.4f" %(average_val_recall))
        print("Average F1 Score: %.4f \n" %(average_val_f1score))
        
    model.save('Project Assets/ablation_DenseNet_Model')
    
def ablation_test_VGG(x_test, y_test):
    """
    The main VGG testing function.
    The function calls the trained model of the VGG that was saved locally, and uses it to test on testing data.
    Results will include an accuracy score, confusion matrix, and a report regarding detailed accuracy, precision, recall, and f1 score
    
    Key Parameters:
    x_test = the testing set used for testing the trained VGG model, must already be processed by the image_processing function
    y_test = supervised labels for  the testing set used for testing the trained VGG model
    
    
    Returns:
    Predicted results for testing data
    Performance metrics of model for testing data 
    """
    model = keras.models.load_model('Project Assets/ablation_VGG_Model')
    ohe = joblib.load('Project Assets/ohe_encoder.joblib')

    result=model.predict(x_test)
    result_class = tf.one_hot(np.argmax(result, axis=1), depth = 3)

    result_class = ohe.inverse_transform(result_class)
    y_test_class = ohe.inverse_transform(y_test)

    acc = accuracy_score(result_class, y_test_class)
    print("Accuracy for test data:", acc)

    plt.figure(figsize = (7,7))
    ConfusionMatrixDisplay.from_predictions(y_test_class, result_class, cmap = 'Blues')
    plt.xticks(rotation=45)
    plt.show()
    print(classification_report(y_test_class, result_class))
    
def ablation_test_densenet(x_test, y_test):
    """
    The main DenseNet testing function.
    The function calls the trained model of the DenseNet that was saved locally, and uses it to test on testing data.
    Results will include an accuracy score, confusion matrix, and a report regarding detailed accuracy, precision, recall, and f1 score
    
    Key Parameters:
    x_test = the testing set used for testing the trained DenseNet model, must already be processed by the image_processing function
    y_test = supervised labels for  the testing set used for testing the trained DenseNet model
    
    
    Returns:
    Predicted results for testing data
    Performance metrics of model for testing data 
    """
    model = keras.models.load_model('Project Assets/ablation_DenseNet_Model')
    ohe = joblib.load('Project Assets/ohe_encoder.joblib')

    result=model.predict(x_test)
    result_class = tf.one_hot(np.argmax(result, axis=1), depth = 3)

    result_class = ohe.inverse_transform(result_class)
    y_test_class = ohe.inverse_transform(y_test)

    acc = accuracy_score(result_class, y_test_class)
    print("Accuracy for test data:", acc)

    plt.figure(figsize = (7,7))
    ConfusionMatrixDisplay.from_predictions(y_test_class, result_class, cmap = 'Blues')
    plt.xticks(rotation=45)
    plt.show()
    print(classification_report(y_test_class, result_class))