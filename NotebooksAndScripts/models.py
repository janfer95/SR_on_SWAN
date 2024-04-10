"""Module that contains the different neural network architectures.

Contains specifically the DSC/MS model, the corresponding surrogate model, and
the FCNN and CNN models for comparison.
"""

from keras.layers import (
    Input,
    Conv2D,
    Dense,
    Concatenate,
    MaxPooling2D,
    UpSampling2D,
    Flatten,
    Reshape,
)
from keras.models import Model

from functools import partial


Conv = partial(Conv2D, activation="relu", padding="same")


def dscms(
    grid: tuple[int, int] = (10, 10),
    upfactor: int = 16,
    dim: int = 1
) -> Model:
    """Initialize the DSC/MS model.

    Parameters
    ----------
    grid : tuple[int, int]
        The grid size of the low-resolution input image.
    upfactor : int
        The upscaling factor.
    dim : int
        The number of channels in the input image.

    Returns
    -------
    Model
        The compiled DSC/MS model.
    """
    input_img = Input(shape=(grid[0], grid[1], dim))
    input_img_up = UpSampling2D((upfactor, upfactor))(input_img)

    down_1 = MaxPooling2D((8, 8))(input_img_up)
    x1 = Conv(32, (3, 3))(down_1)
    x1 = Conv(32, (3, 3))(x1)
    x1 = UpSampling2D((2, 2))(x1)

    down_2 = MaxPooling2D((4, 4))(input_img_up)
    x2 = Concatenate()([x1, down_2])
    x2 = Conv(32, (3, 3))(x2)
    x2 = Conv(32, (3, 3))(x2)
    x2 = UpSampling2D((2, 2))(x2)

    down_3 = MaxPooling2D((2, 2))(input_img_up)
    x3 = Concatenate()([x2, down_3])
    x3 = Conv(32, (3, 3))(x3)
    x3 = Conv(32, (3, 3))(x3)
    x3 = UpSampling2D((2, 2))(x3)

    x4 = Concatenate()([x3, input_img_up])
    x4 = Conv(32, (3, 3))(x4)
    x4 = Conv(32, (3, 3))(x4)

    # Multi-scale model (Du et al., 2018)
    layer_1 = Conv(16, (5, 5))(input_img_up)
    x1m = Conv(8, (5, 5))(layer_1)
    x1m = Conv(8, (5, 5))(x1m)

    layer_2 = Conv(16, (9, 9))(input_img_up)
    x2m = Conv(8, (9, 9))(layer_2)
    x2m = Conv(8, (9, 9))(x2m)

    layer_3 = Conv(16, (13, 13))(input_img_up)
    x3m = Conv(8, (13, 13))(layer_3)
    x3m = Conv(8, (13, 13))(x3m)

    x_add = Concatenate()([x1m, x2m, x3m, input_img_up])
    x4m = Conv(8, (7, 7))(x_add)
    x4m = Conv(3, (5, 5))(x4m)

    x_final = Concatenate()([x4, x4m])
    x_final = Conv2D(dim, (3, 3), activation=None, padding="same")(x_final)
    model = Model(input_img, x_final)
    model.compile(optimizer="adam", loss="mae")

    return model


def surrogate(
    grid: tuple[int, int] = (160, 160),
    nfreq: int = 32,
    ntheta: int = 24,
    dim: int = 1
) -> Model:
    """Initialize the surrogate model.

    Parameters
    ----------
    grid : tuple[int, int]
        The grid size of the input image (high-resolution bathymetry
        in our case).
    nfreq : int
        The number of frequency bins.
    ntheta : int
        The number of directional bins.
    dim : int
        The number of channels in the input image.
    """
    input_bathy = Input(shape=(grid[0], grid[1], dim))
    input_spec = Input(shape=(nfreq, ntheta, dim))

    flat_spec = Flatten()(input_spec)
    flat_spec = Dense(nfreq * ntheta, activation="relu")(flat_spec)
    flat_spec = Dense(grid[0] * grid[1], activation="relu")(flat_spec)

    spec = Reshape((grid[0], grid[1], dim))(flat_spec)
    input_img_up = Concatenate()([spec, input_bathy])

    down_1 = MaxPooling2D((8, 8), padding="same")(input_img_up)
    x1 = Conv(32, (3, 3))(down_1)
    x1 = Conv(32, (3, 3))(x1)
    x1 = UpSampling2D((2, 2))(x1)

    down_2 = MaxPooling2D((4, 4), padding="same")(input_img_up)
    x2 = Concatenate()([x1, down_2])
    x2 = Conv(32, (3, 3))(x2)
    x2 = Conv(32, (3, 3))(x2)
    x2 = UpSampling2D((2, 2))(x2)

    down_3 = MaxPooling2D((2, 2), padding="same")(input_img_up)
    x3 = Concatenate()([x2, down_3])
    x3 = Conv(32, (3, 3))(x3)
    x3 = Conv(32, (3, 3))(x3)
    x3 = UpSampling2D((2, 2))(x3)

    x4 = Concatenate()([x3, input_img_up])
    x4 = Conv(32, (3, 3))(x4)
    x4 = Conv(32, (3, 3))(x4)

    # Multi-scale model (Du et al., 2018)
    layer_1 = Conv(16, (5, 5))(input_img_up)
    x1m = Conv(8, (5, 5))(layer_1)
    x1m = Conv(8, (5, 5))(x1m)

    layer_2 = Conv(16, (9, 9))(input_img_up)
    x2m = Conv(8, (9, 9))(layer_2)
    x2m = Conv(8, (9, 9))(x2m)

    layer_3 = Conv(16, (13, 13))(input_img_up)
    x3m = Conv(8, (13, 13))(layer_3)
    x3m = Conv(8, (13, 13))(x3m)

    x_add = Concatenate()([x1m, x2m, x3m, input_img_up])
    x4m = Conv(8, (7, 7))(x_add)
    x4m = Conv(3, (5, 5))(x4m)

    x_final = Concatenate()([x4, x4m])
    x_final = Conv2D(dim, (3, 3), activation=None, padding="same")(x_final)
    model = Model([input_spec, input_bathy], x_final)
    model.compile(optimizer="adam", loss="mae")

    return model


def fcnn(
    grid: tuple[int, int] = (10, 10),
    upfactor: int = 16,
    dim: int = 1
) -> Model:
    """Initialize the Fully-Connected Neural Network model.

    Parameters
    ----------
    grid : tuple[int, int]
        The grid size of the low-resolution input image.
    upfactor : int
        The upscaling factor.
    dim : int
        The number of channels in the input image.

    Returns
    -------
    Model
        The compiled FCNN model.
    """
    input_img = Input(shape=(grid[0], grid[1], dim))

    x = Flatten()(input_img)

    x = Dense(3000, activation="relu")(x)
    x = Dense(10000, activation="relu")(x)
    x = Dense(grid[0] * grid[1] * upfactor**2, activation=None)(x)
    x_final = Reshape((grid[0] * upfactor, grid[1] * upfactor))

    model = Model(input_img, x_final)
    model.compile(optimizer="adam", loss="mae")

    return model


def cnn(
    grid: tuple[int, int] = (10, 10),
    upfactor: int = 16,
    dim: int = 1
):
    """Initialize the Convolutional Neural Network model.

    Parameters
    ----------
    grid : tuple[int, int]
        The grid size of the low-resolution input image.
    upfactor : int
        The upscaling factor.
    dim : int
        The number of channels in the input image.

    Returns
    -------
    Model
        The compiled CNN model.
    """
    input_img = Input(shape=(grid[0], grid[1], dim))
    input_img_up = UpSampling2D((upfactor, upfactor))(input_img)

    x = Conv(32, (3, 3))(input_img_up)
    x = Conv(32, (3, 3))(x)
    x = Conv(32, (3, 3))(x)
    x_final = Conv2D(dim, (3, 3), activation=None, padding="same")(x)

    model = Model(input_img, x_final)
    model.compile(optimizer="adam", loss="mae")

    return model
