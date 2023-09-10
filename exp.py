"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import data_preprocess, train_model, read_digits, split_train_dev_test, p_and_eval, hparams_tune
import itertools
from itertools import permutations
import pdb

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.


# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

## Split data 
X, y = read_digits()


gammas = [0.1, 0.001, 0.005, 0.123, 0.879, 0.0009]
cparams = [0.3, 1, 2, 0.7, 2, 8, 7, 0.9]

p_comb = {
    'gammas' : gammas,
    'cparams' : cparams
}

test_size = [0.1, 0.2, 0.3]
dev_size = [0.1, 0.2, 0.3]

for tsize in test_size:
    for dsize in dev_size:
        # X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.3)
        X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(X, y, test_size=tsize, dev_size=dsize)

        ## Use the preprocessed datas
        X_train = data_preprocess(X_train)
        X_test = data_preprocess(X_test)
        X_dev = data_preprocess(X_dev)

        # Calling hparams function to test with different hyperparameters
        best_params, best_model, best_accur = hparams_tune(X_train, X_dev, y_train, y_dev, p_comb)

        model = train_model(X_train, y_train, {'gamma': best_params[0], 'C' : best_params[1]}, model_type='svm')
        # Predict the value of the digit on the test subset

        # Get the test accuracy 
        test_accur = p_and_eval(model, X_test, y_test)

        print(f"test_size={tsize} dev_size={dsize} train_size={1 - (tsize + dsize)} train_acc={best_accur:.2f} dev_acc={best_accur:.2f} test_acc={test_accur:.2f}")
        print(f"Best Hyperparameters: ( gamma : {best_params[0]} , C : {best_params[1]} )\n")

