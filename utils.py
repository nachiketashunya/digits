# Import datasets, classifiers and performance metrics
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
import matplotlib.pyplot as plt
import itertools
from sklearn import datasets, metrics, svm , tree
from sklearn.model_selection import train_test_split
from joblib import dump,load
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import numpy as np


def read_digits():
    data = datasets.load_digits()
    X = data.images
    y = data.target
    return X, y

## function for data preprocessing
def data_preprocess(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data 


def resize_images(data, height, width):
    # Resize the images
    resized_images = []
    for image in data:
        resized_image = resize(image, (height, width), anti_aliasing=True)
        resized_images.append(resized_image)

    # Convert the list of resized images back to a NumPy array
    data = np.array(resized_images)

    return data

def get_all_h_param_comb_svm(gamma_list,c_list):
    return list(itertools.product(gamma_list, c_list))

def get_all_h_param_comb_tree(depth_list):
    return list(itertools.product(depth_list))
 
## Function for splitting data
def split_dataset(X, y, test_size, random_state = 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False, random_state=random_state)

    return X_train, X_test, y_train, y_test 


## Function for training model
def train_model(x, y, model_params, model_type='svm'):
    if model_type == 'svm':
        clf = svm.SVC
    if model_type=='tree':
        clf = tree.DecisionTreeClassifier
    model = clf(**model_params)
    # pdb.set_trace()
    model.fit(x, y)
    return model 


def split_train_dev_test(X, y, test_size, dev_size):
    # Split data into test and temporary (train + dev) sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
    
    # Calculate the ratio between dev and temp sizes
    dev_ratio = dev_size / (1 - test_size)
    
    # Split temporary data into train and dev sets
    X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=dev_ratio, shuffle=True)
    
    return X_train, X_test, X_dev, y_train, y_test, y_dev

def p_and_eval(model, metric, X_test, y_test):
    predicted = model.predict(X_test)
    accuracy = metric(y_pred=predicted, y_true=y_test)
    return accuracy  

# Function for hyperparameter tunning
def hparams_tune(X_train, y_train, X_dev, y_dev, all_combos,metric,model_type='svm'):
    best_accuracy = -1
    best_model=None
    best_hparams = None
    best_model_path=""

    for param in all_combos:
        if model_type=="Production_Model_svm":
            cur_model = train_model(X_train,y_train,{'gamma':param[0],'C':param[1]},model_type='svm')
        if model_type=="Candidate_Model_tree":
            cur_model = train_model(X_train,y_train,{'max_depth':param[0]},model_type='tree')    
        val_accuracy = p_and_eval(cur_model,metric,X_dev,y_dev)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_hparams=param
            best_model_path = "./models/{}{}.joblib".format(model_type, param).replace(":", "")
            best_model = cur_model
        
    dump(best_model,best_model_path) 
    # print("Model save at {}".format(best_model_path))   
    return best_hparams, best_model_path, best_accuracy 
