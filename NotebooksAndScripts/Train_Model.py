"""This is a script to train a neural network
with an architecture based on the Downsampled-Skip
Connection / Multi-Scale model introduced by
Fukami et al. 2019.
"""
import os
import argparse
import models
import numpy as np
import tensorflow as tf
import utilities as u

from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from time import time


# Ignore Tensorflow Warnings and other tensorflow options
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Parse variable parameter, for easier training
parser = argparse.ArgumentParser(description="Process parameters")
parser.add_argument("model", metavar="m", type=str, nargs='?',
                    default="Subpixel", help="Which model to train")
parser.add_argument("variable", metavar="v", type=str, nargs='?',
                    default="Hs", help="Which variable to use")
args = parser.parse_args()


# Setting for Training datasets
model = args.model
var = args.variable
upfactor = 16
grid = (10, 10)
grid_out = (grid[0]*upfactor, grid[1]*upfactor)

# Beginning and end of the data set serial
sample_start = 24
sample_end = 8760
serial = np.arange(sample_start, sample_end+1, 1)

# Beginning and end of test data serial
sample_start_test = 8761
sample_end_test = 17496
serial_test = np.arange(sample_start_test, sample_end_test+1, 1)

# Learning parameters
train_size = 0.8
patience = 30
batch_size = 32

# Thresholds for Data Augmentation
thresholds = {"Hs": 4, "Tm02": 12, "Dir": 330}

# Prepare file locations for reference and input data
fname_HR = './Data/HR/{}/BaskCoast_{}_{{}}.npy'
fname_HR = fname_HR.format(var, var.upper())
fname_LR = './Data/LR/{}/BaskCoast_{}_{{}}.npy'
fname_LR = fname_LR.format(var, var.upper())

fnames = [fname_HR, fname_LR]

# Load training data
X_tot, y_tot = u.load_data(fnames, serial, var)

# Load thresholded pool data
if model == "DataAugmented":
    fname_pool = './Data/LR/AvgPool/{}/BaskCoast_{}_{{}}.npy'
    fname_pool = fname_pool.format(upfactor, var, var.upper())
    fnames_pool = [fname_HR, fname_pool]
    X_pool, y_pool = u.load_threshold_data(thresholds, fnames_pool,
                                           serial, var)

# Create random integer to set a random state for train_test_split
random_state_train, random_state_test = np.random.randint(1e6, size=2)

# Divide into training and validation data
X_train, X_valid, y_train, y_valid = train_test_split(
    X_tot, y_tot, train_size=train_size,
    random_state=random_state_train)

if model == "DataAugmented":
    # Concat the pool data to the training set
    X_train = np.r_[X_train, X_pool]
    y_train = np.r_[y_train, y_pool]

# Extra shuffle data
np.random.seed(random_state_train)
np.random.shuffle(X_train)

np.random.seed(random_state_train)
np.random.shuffle(y_train)

# Free up some resources
del X_tot, y_tot

# Set all nans to zero
X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)

X_valid = np.nan_to_num(X_valid)
y_valid = np.nan_to_num(y_valid)


print("X_train shape: ", X_train.shape)
print("X_valid shape: ", X_valid.shape)

print("Initializing Model")

# Choose the right model
if model == "Subpixel" or model == "DataAugmented":
    autoencoder = models.Subpixel(grid=(10, 10))
elif model == "SubpixelDilated":
    autoencoder = models.SubpixelDilated(grid=(10, 10))
else:
    autoencoder = models.OriginalFukami(grid=(10, 10))

print(autoencoder.summary())

# Define file output names
fdir = "Models/{}".format(model)
fmodel, fhist, fsum = u.get_info_file_names(fdir, var)

# Save only best models
model_cb = ModelCheckpoint(fmodel, monitor='val_loss',
                           save_best_only=True, verbose=1)
# Implement Early Stopping
early_cb = EarlyStopping(monitor='val_loss', patience=patience,
                         verbose=1)
cb = [model_cb, early_cb]

# Train Neural Network and measure how long it takes
t0 = time()
history = autoencoder.fit(X_train, y_train, epochs=5000,
                          batch_size=batch_size, verbose=1,
                          callbacks=cb,
                          validation_data=(X_valid, y_valid))
t0 = time()-t0

# Free some resources for test data evaluation
del X_train, y_train, X_valid, y_valid

# Load test data and set it to nan
X_test, y_test = u.load_data(fnames, serial_test, var)

X_test = np.nan_to_num(X_test)
y_test = np.nan_to_num(y_test)

# Evaluate test data
ev = autoencoder.evaluate(X_test, y_test)

# Save Model History
d = history.history
header = "Loss, Val_Loss"
np.savetxt(fhist, np.c_[d["loss"], d["val_loss"]], header=header,
           delimiter=",")

# Get the best val_loss for the summary file
min_val_loss = min(d["val_loss"])

u.write_summary(fsum, var, t0, random_state_train,
                random_state_test, min_val_loss, ev)