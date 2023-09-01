# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

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