"""This is a script to train a neural network
with an architecture based on the Downsampled-Skip
Connection / Multi-Scale model introduced by
Fukami et al. 2019.
"""
import os
import argparse
import numpy as np
import tensorflow as tf
import utilities as u

from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from time import time
from pathlib import Path

from models import Surrogate


# Ignore Tensorflow Warnings and other tensorflow options
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Parse variable parameter, for easier training
parser = argparse.ArgumentParser(description="Process parameters")
parser.add_argument("variable", metavar="v", type=str, nargs='?',
                    default="Hs", help="Which variable to use")
args = parser.parse_args()

model = "Surrogate"

# Setting for Training datasets
var = args.variable
nfreq = 32
ntheta = 24
grid = (32, 24)
grid_out = (160, 160)
dim = 1

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

# Prepare file locations for reference and input data
fname_HR = f'Data/HR/{var}/BaskCoast_{var.upper()}_{{}}.npy'
fname_LR = 'Data/Spectrum/BaskCoast_{}.npy'
fname_bat = './Data/Bathymetry/bat.npy'

fnames = [fname_HR, fname_LR]

# Load training data and shift directional data if needed
X_tot, y_tot = u.load_data(fnames, serial, var, convert=False, grid=grid)
if var == "Dir":
    y_tot = (y_tot - 255 + 360) % 360

# Load bathymetry and tile to right length
bat_tot = np.load(fname_bat).reshape((1, *grid_out, dim))
bat_tot = np.tile(bat_tot, (len(serial), 1, 1, 1))

# Compute normalization constants
X_max, X_min = np.nanmax(X_tot), np.nanmin(X_tot)
bat_max, bat_min = np.nanmax(bat_tot), np.nanmin(bat_tot)

# Save for later use
np.save("Data/Xmax_Xmin_2018_spectrum.npy", np.array([X_max, X_min]))
np.save("Data/bat_max_bat_min_2018_spectrum.npy", np.array([bat_max, bat_min]))

# Normalize data
X_tot = (X_tot - X_min) / (X_max - X_min)
bat_tot = (bat_tot - bat_min) / (bat_max - bat_min)

# Create random integer to set a random state for train_test_split
random_state_train, random_state_test = np.random.randint(1e6, size=2)

# Divide into training and validation data
X_train, X_valid, y_train, y_valid = train_test_split(
    X_tot, y_tot, train_size=train_size,
    random_state=random_state_train)

idx = int(0.8*X_tot.shape[0])
bat_train = bat_tot[:idx]
bat_valid = bat_tot[idx:]

# Free up some resources
del X_tot, y_tot, bat_tot

# Set all nans to zero
np.nan_to_num(X_train, copy=False)
np.nan_to_num(y_train, copy=False)

np.nan_to_num(X_valid, copy=False)
np.nan_to_num(y_valid, copy=False)

print("X_train shape: ", X_train.shape)
print("X_valid shape: ", X_valid.shape)
print("Bat_train shape:", bat_train.shape)

print("Initializing Model")

# Choose the right model
autoencoder = Surrogate(grid_out, nfreq, ntheta, dim)

print(autoencoder.summary())

# Define file output names
fdir = f"Models/{model}"
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
history = autoencoder.fit([X_train, bat_train], y_train, epochs=5000,
                          batch_size=batch_size, verbose=1,
                          callbacks=cb,
                          validation_data=([X_valid, bat_valid], y_valid))
t0 = time()-t0

# Free some resources for test data evaluation
del X_train, y_train, X_valid, y_valid, bat_train, bat_valid

# Load test data and set it to nan
X_test, y_test = u.load_data(fnames, serial_test, var, convert=False)

# Load and tile bathymetry data
bat_test = np.load(fname_bat).reshape((1, *grid_out, dim))
bat_test = np.tile(bat_test, (len(serial_test), 1, 1, 1))

X_test = (X_test - X_min) / (X_max - X_min)
bat_test = (bat_test - bat_min) / (bat_max - bat_min)

if var == "Dir":
    y_test = (y_test - 255 + 360) % 360

X_test = np.nan_to_num(X_test)
y_test = np.nan_to_num(y_test)

# Evaluate test data
ev = autoencoder.evaluate([X_test, bat_test], y_test)

# Save Model History
d = history.history
header = "Loss, Val_Loss"
np.savetxt(fhist, np.c_[d["loss"], d["val_loss"]], header=header,
           delimiter=",")

# Get the best val_loss for the summary file
min_val_loss = min(d["val_loss"])

u.write_summary(fsum, var, t0, random_state_train,
                random_state_test, min_val_loss, ev)
