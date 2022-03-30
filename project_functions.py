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
from sklearn import svm
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

def image_processing(data_path,file_names,img_size):
    dataset_tumor = []
    for file_name in file_names:
        file=cv2.imread(data_path+file_name+'.png', cv2.IMREAD_COLOR) 
        file_resize=cv2.resize(file,(img_size,img_size))/255.
        dataset_tumor.append(file_resize)
    tumor_data = np.array(dataset_tumor)
    tumor_data = tumor_data.reshape(-1,img_size,img_size,3)
    return tumor_data

def preprocessing_data(data_path, file, image_size):
    data = pd.read_csv(file)
    data = data.replace({0:"No DR", 1:"Mild DR", 2:"Mild DR", 3:"Clear DR",4:"Clear DR"})
    output_nodes = len(set(list(data['diagnosis'])))
    file_names=list(data['id_code'])
    labels=data['diagnosis'].values.ravel()
    #x = image_processing(data_path,file_names,image_size)
    #np.save('Project Assets/x_image_data', x)
    x = np.load('Project Assets/x_image_data.npy')
    x_train_val,x_test,y_train_val,y_test = train_test_split(x,labels,test_size=0.2)
    
    ohe = OneHotEncoder(handle_unknown = "ignore", sparse=False)
    ohe = ohe.fit(labels.reshape(-1,1))
    y_train_val = ohe.transform(np.array(y_train_val).reshape(-1,1))
    y_test = ohe.transform(np.array(y_test).reshape(-1,1))
    dump(ohe, 'Project Assets/ohe_encoder.joblib') 
    
    return x_train_val,x_test,y_train_val,y_test
    
def build_VGG(image_size):
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
    
    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    
    model.add(Dense(3, activation='softmax'))
    model.compile(
        loss='categorical_hinge',
        optimizer=Adam(learning_rate=0.001),
        metrics=['acc']
    )
    
    return model

def build_densenet(image_size):
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
    #model.add(Dropout(0.5))
    
    model.add(Dense(3, activation='softmax'))
    model.compile(
        loss='categorical_hinge',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['acc']
    )
    
    return model

def train_VGG(x_train_val, y_train_val, image_size):

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
        
        model = build_VGG(image_size)
        
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
        
    model.save('Project Assets/VGG_Model')
    
def train_densenet(x_train_val, y_train_val, image_size):

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
        
        model = build_densenet(image_size)
        
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
        
    model.save('Project Assets/DenseNet_Model')
    
def test_VGG(x_test, y_test):
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
    model = keras.models.load_model('Project Assets/VGG_Model')
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
    
def test_densenet(x_test, y_test):
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
    model = keras.models.load_model('Project Assets/DenseNet_Model')
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