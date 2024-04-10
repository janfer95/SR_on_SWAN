"""Various utility functions that are used repeatedly throughout the
jupyter notebooks and model scripts, such as for loading data or writing
and working with summary files.
"""

import numpy as np

from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal, Union


PathLike = Union[str, Path]


def uniquify(paths: list[Path]) -> list[Path]:
    """Adds a counter to the end of the filename if it already exists.

    Parameters
    ----------
    paths : list[Path]
        List of paths to files that need to be uniquified

    Returns
    -------
    list[Path]
        List of paths to files with the added counter
    """
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


def create_dategenerator(
    time_step: int = 1,
    time_step_type: Literal["hrs", "min"] = "hrs"
) -> Callable[[date, date], Iterator[date]]:
    """Create a generator that yields a date range with a given time step.

    Parameters
    ----------
    time_step : int, optional
        Time step in hours or minutes, by default 1
    time_step_type : Literal["hrs", "min"], optional
        Type of time step, by default "hrs"

    Returns
    -------
    Callable[[date, date], Iterator[date]]
        Generator that yields a date range with the given time step
    """
    if time_step_type == "hrs":
        const = int(24 / time_step)

        def daterange(start_date, end_date):
            for n in range(const * int((end_date - start_date).days)):
                yield start_date + n * timedelta(hours=time_step)

    elif time_step_type == "min":
        const = int(60 / time_step * 24)

        def daterange(start_date, end_date):
            for n in range(const * int((end_date - start_date).days)):
                yield start_date + n * timedelta(minutes=time_step)
    else:
        raise ValueError("time_step_type must be 'hrs' or 'min'")

    return daterange


def get_dict(
    filename: PathLike
) -> dict[str, str]:
    """Get the summary file as a dictionary.

    Parameters
    ----------
    filename : PathLike
        Path to the summary file

    Returns
    -------
    dict[str, str]
        Dictionary with the summary file contents
    """
    with open(filename) as f:
        # Get everything into a dict
        d = dict(line.split(": ", 1) for line in f)

    return d


def get_file_list(
    dr: PathLike,
    var: str,
    root_dir: PathLike = "Models",
    file_prefix: str = "Summary_{}*"
) -> list[Path]:
    """Get a list of summary files, given a model directory and a variable.

    Parameters
    ----------
    dr : PathLike
        Model directory
    var : str
        Variable name.
    root_dir : PathLike, optional
        Root directory of the models, by default "Models"
    file_prefix : str, optional
        File prefix for the summary files, by default "Summary_{}*"

    Returns
    -------
    list[Path]
        List of summary files in the given directory
    """

    fdir = Path(root_dir) / dr

    return fdir.glob(file_prefix.format(var))


def extract_time_and_eval(
    dr: PathLike,
    var: str,
    root_dir: PathLike = "Models",
    file_prefix: str = "Summary_{}*"
) -> tuple[list[float], list[float]]:
    """Extract the run time and the final evaluation loss of the model

    Note that the time is in hours.

    Parameters
    ----------
    dr : PathLike
        Model directory
    var : str
        Variable name.
    root_dir : PathLike, optional
        Root directory of the models, by default "Models"
    file_prefix : str, optional
        File prefix for the summary files, by default "Summary_{}*"

    Returns
    -------
    tuple[list[float], list[float]]
        List of evaluation losses and run times
    """
    flist = get_file_list(dr, var, root_dir, file_prefix)

    times = []
    evals = []

    for filename in flist:
        d = get_dict(filename)
        time = float(d["Time"]) / 3600
        ev = float(d["Eval"])

        times.append(time)
        evals.append(ev)

    return evals, times


def write_summary(
    fsum: PathLike,
    var: str,
    t0: float,
    random_state_train: int,
    random_state_test: int,
    min_val_loss: float,
    ev: float,
    train_size: float = 0.8,
    batch_size: int = 32,
    patience: int = 30,
) -> None:
    """Write the summary file for the model.

    Parameters
    ----------
    fsum : PathLike
        Path to the summary file
    var : str
        Variable name
    t0 : float
        Run time of the model
    random_state_train : int
        Random state for the training set
    random_state_test : int
        Random state for the test set
    min_val_loss : float
        Minimum validation loss
    ev : float
        Evaluation / Test loss
    train_size : float, optional
        Fraction of data set used for training, by default 0.8
    batch_size : int, optional
        Batch size, by default 32
    patience : int, optional
        Patience for early stopping, by default 30

    Returns
    -------
    None
    """
    summary = """Date: {today}
    Variable: {var}
    Train_Size: {train_size}
    Patience: {patience}
    Batch_Size: {batch_size}
    Time: {t0}
    Random_State_Train: {random_state_train}
    Random_State_Test: {random_state_test}
    min_val_loss: {min_val_loss}
    Eval: {ev}
    """

    # Get the current date
    today = date.today().strftime("%Y%m%d")

    with open(fsum, "w") as f:
        summary = summary.format(
            today=today,
            var=var,
            train_size=train_size,
            patience=patience,
            batch_size=batch_size,
            t0=t0,
            random_state_train=random_state_train,
            random_state_test=random_state_test,
            min_val_loss=min_val_loss,
            ev=ev,
        )
        f.write(summary)


def get_info_file_names(fdir: PathLike, var: str) -> tuple[Path, Path, Path]:
    """Create and return the paths to the information files.

    If necessary the filenames are made unique by appending a suffix counter.

    Parameters
    ----------
    fdir : PathLike
        Model directory to save the files in
    var : str
        Variable name

    Returns
    -------
    tuple[Path, Path, Path]
        Paths to the model, history, and summary files
    """
    fdir = Path(fdir)

    fdir.mkdir(parents=True, exist_ok=True)

    # Get the current date to append to the model name
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


def load_data(
    fnames: tuple[str, str],
    serial: Iterable[int],
    var: str,
    grid: tuple[int, int] = (10, 10),
    grid_out: tuple[int, int] = (160, 160),
    dim: int = 1,
    convert: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Load the data from the given filename patterns and serial numbers.

    Parameters
    ----------
    fnames : tuple[str, str]
        Tuple of filename patterns for the high and low resolution data
    serial : Iterable[int]
        Serial numbers of the data to load
    var : str
        Variable name
    grid : tuple[int, int], optional
        Grid size of the input data, by default (10, 10)
    grid_out : tuple[int, int], optional
        Grid size of the output data, by default (160, 160)
    dim : int, optional
        Number of grid dimensions. Can be useful for multiple input types,
        such as wave height AND wave period. By default 1
    convert : bool, optional
        Convert directional data to a range without the 360°-0° discontinuity,
        by default False

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of low-resolution input and high-resolution output data
    """
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


def NNUpsampling(arr: np.ndarray, factor: int = 16) -> np.ndarray:
    """Simple implementation of nearest neighbor upsampling in numpy.

    Parameters
    ----------
    arr : np.ndarray
        Array to upsample
    factor : int, optional
        Upsampling factor, by default 16

    Returns
    -------
    np.ndarray
        Upsampled array
    """
    return arr.repeat(factor, axis=0).repeat(factor, axis=1)


def get_data_in_box(
    coords: np.ndarray,
    bathy: np.ndarray,
    tr: tuple[float, float],
    bl: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the bathymetry data and coordinates within the defined box.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of the data
    bathy : np.ndarray
        Bathymetry data
    tr : tuple[float, float]
        Top right corner of the box
    bl : tuple[float, float]
        Bottom left corner of the box

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Coordinates and bathymetry data within the box
    """
    condition1 = coords[:, 0] > bl[0]
    condition2 = coords[:, 1] > bl[1]
    condition3 = coords[:, 0] < tr[0]
    condition4 = coords[:, 1] < tr[1]

    condition = condition1 & condition2 & condition3 & condition4

    return coords[condition], bathy[condition]
