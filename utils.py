# Import datasets, classifiers and performance metrics
import matplotlib.pyplot as plt

from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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

 
## Function for splitting data
def split_dataset(X, y, test_size, random_state = 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False, random_state=random_state)

    return X_train, X_test, y_train, y_test 


## Function for training model
def train_model(x, y, model_params, model_type='svm'):
    if model_type == 'svm':
        clf = svm.SVC

    model = clf(**model_params)
    # pdb.set_trace()
    model.fit(x, y)
    return model 


def split_train_dev_test(X, y, test_size, dev_size):
    # Split data into test and temporary (train + dev) sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Calculate the ratio between dev and temp sizes
    dev_ratio = dev_size / (1 - test_size)
    
    # Split temporary data into train and dev sets
    X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=dev_ratio, shuffle=False)
    
    return X_train, X_test, X_dev, y_train, y_test, y_dev

def p_and_eval(model, X_test, y_test):
    # Predict the values using the model
    predicted = model.predict(X_test)    
    return metrics.accuracy_score(y_test, predicted)