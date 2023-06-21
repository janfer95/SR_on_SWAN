"""This module contains the subpixel layer class and
various other often-used functions to keep the
training scripts and jupyter notebooks tidy."""
import numpy as np
import glob

from datetime import date, timedelta
from pathlib import Path


# Utility Functions
def uniquify(paths):
    """If model name already exists, add a subscript."""
    paths_out = []
    for path in paths:
        path = Path(path)
        counter = 1
        stem = path.stem

        while path.is_file():
            path = path.with_stem(stem + f"_{counter}")
            counter += 1

        paths_out.append(path)
    return paths_out


def create_dategenerator(time_step=1, time_step_type="hrs"):
    """Define help function to create a generator that outputs a
    datetime object for the given time range."""
    if time_step_type == "hrs":
        const = int(24/time_step)

        def daterange(start_date, end_date):
            for n in range(const*int((end_date - start_date).days)):
                yield start_date + n*timedelta(hours=time_step)

    else:
        const = int(60/time_step * 24)

        def daterange(start_date, end_date):
            for n in range(const*int((end_date - start_date).days)):
                yield start_date + n*timedelta(minutes=time_step)

    return daterange


def get_dict(filename):
    """
    Get the summary file as a dictionary.
    """
    with open(filename) as f:
        # Get everything into a dict
        d = dict(line.split(': ', 1) for line in f)

    return d


def get_file_list(dr, var, dir_prefix="Models",
                  file_prefix="Summary_{}*"):
    """
    Get a list of summary files, given a model directory and a variable.
    """
    
    fdir = Path(dir_prefix) / dr

    return fdir.glob(file_prefix.format(var))


def extract_time_and_eval(dr, var, **kwargs):
    """
    Extract the run time and the final evaluation of the model,
    given a model directory and a variable. Convert times into hours.
    """
    # Get the file list
    flist = get_file_list(dr, var, **kwargs)

    times = []
    evals = []

    for filename in flist:
        d = get_dict(filename)
        time = float(d["Time"])/3600
        ev = float(d["Eval"])

        times.append(time)
        evals.append(ev)

    return evals, times


def write_summary(fsum, var, t0, random_state_train,
                  random_state_test, min_val_loss, ev,
                  train_size=0.8, batch_size=32,
                  patience=30):
    """Write the summary file."""
    summary = """Date: {}
    Variable: {}
    Train_Size: {}
    Patience: {}
    Batch_Size: {}
    Time: {}
    Random_State_Train: {}
    Random_State_Test: {}
    min_val_loss: {}
    Eval: {}
    """

    # Get the current date
    today = date.today().strftime("%Y%m%d")

    with open(fsum, "w") as f:
        summary = summary.format(today, var, train_size, patience,
                                 batch_size, t0, random_state_train,
                                 random_state_test, min_val_loss, ev)
        f.write(summary)


def get_info_file_names(fdir, var):
    """Return the paths to the information files."""
    fdir = Path(fdir)
    
    # Check if Model Folder already exists, if not, create it
    if not fdir.exists():
        fdir.mkdir(parents=True)

    # Get the current date for the model name
    today = date.today().strftime("%Y%m%d")

    # Define the model names
    fmodel = f"Model_Inp_{var}_{today}.hdf5"
    fmodel = fdir / fmodel

    fhist = f"History_Inp_{var}_{today}.csv"
    fhist = fdir / fhist

    fsum = f"Summary_Inp_{var}_{today}.txt"
    fsum = fdir / fsum

    # Check if filename already exists and if it does uniquify it
    if fmodel.is_file():
        fmodel, fhist, fsum = uniquify([fmodel, fhist, fsum])

    return fmodel, fhist, fsum


# Data Handling
def load_data(fnames, serial, var, grid=(10, 10), grid_out=(160, 160),
              dim=1, convert=True):
    """Load input and reference data."""
    fname_HR, fname_LR = fnames
    n_serial = len(serial)
    # Initialize input and labels arrays respectively
    X_tot = np.zeros((n_serial, grid[0], grid[1], dim))
    y_tot = np.zeros((n_serial, grid_out[0], grid_out[1], dim))

    print("Loading data...")
    # Iterate through all the datasets and save them in their arrays
    for i, s in enumerate(serial):
        # Load reference data
        y_tot[i, :, :, 0] = np.load(fname_HR.format(s))

        # Load input data
        X_tot[i, :, :, 0] = np.load(fname_LR.format(s))

    if var == "Dir" and convert:
        X_tot = (X_tot - 255 + 360) % 360
        y_tot = (y_tot - 255 + 360) % 360

    return X_tot, y_tot


def NNUpsampling(arr, factor=16):
    """
    Simple implementation of nearest neighbor upsampling
    that works on numpy arrays.
    """

    return arr.repeat(factor, axis=0).repeat(factor, axis=1)


def get_data_in_box(coords, bathy, tr, bl):
    """
    Given the bathymetry data (and the corresponding
    coordinates), extract only the data in the
    rectangle defined by top right (tr) and
    bottom left (bl) point.
    """
    # Define the conditions if coord is in box
    condition1 = coords[:, 0] > bl[0]
    condition2 = coords[:, 1] > bl[1]
    condition3 = coords[:, 0] < tr[0]
    condition4 = coords[:, 1] < tr[1]

    condition = condition1 & condition2 & condition3 & condition4

    return coords[condition], bathy[condition]
