"""Module that contains the DSC/MS and surrogate model
architectures."""

from keras.layers import (Input, Conv2D, Dense, Concatenate, MaxPooling2D,
                          UpSampling2D, Flatten, Reshape)
from keras.models import Model


def DscMs(grid=(10, 10), upfactor=16, dim=1):
    """Initialize the DSC/MS model."""
    input_img = Input(shape=(grid[0], grid[1], dim))
    input_img_up = UpSampling2D((upfactor, upfactor))(input_img)

    down_1 = MaxPooling2D((8, 8))(input_img_up)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(down_1)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x1 = UpSampling2D((2, 2))(x1)

    down_2 = MaxPooling2D((4, 4))(input_img_up)
    x2 = Concatenate()([x1, down_2])
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
    x2 = UpSampling2D((2, 2))(x2)

    down_3 = MaxPooling2D((2, 2))(input_img_up)
    x3 = Concatenate()([x2, down_3])
    x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(x3)
    x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(x3)
    x3 = UpSampling2D((2, 2))(x3)

    x4 = Concatenate()([x3, input_img_up])
    x4 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
    x4 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)

    # Multi-scale model (Du et al., 2018)
    layer_1 = Conv2D(16, (5, 5), activation='relu',
                     padding='same')(input_img_up)
    x1m = Conv2D(8, (5, 5), activation='relu', padding='same')(layer_1)
    x1m = Conv2D(8, (5, 5), activation='relu', padding='same')(x1m)

    layer_2 = Conv2D(16, (9, 9), activation='relu',
                     padding='same')(input_img_up)
    x2m = Conv2D(8, (9, 9), activation='relu', padding='same')(layer_2)
    x2m = Conv2D(8, (9, 9), activation='relu', padding='same')(x2m)

    layer_3 = Conv2D(16, (13, 13), activation='relu',
                     padding='same')(input_img_up)
    x3m = Conv2D(8, (13, 13), activation='relu', padding='same')(layer_3)
    x3m = Conv2D(8, (13, 13), activation='relu', padding='same')(x3m)

    x_add = Concatenate()([x1m, x2m, x3m, input_img_up])
    x4m = Conv2D(8, (7, 7), activation='relu', padding='same')(x_add)
    x4m = Conv2D(3, (5, 5), activation='relu', padding='same')(x4m)

    x_final = Concatenate()([x4, x4m])
    x_final = Conv2D(dim, (3, 3), padding='same')(x_final)
    autoencoder = Model(input_img, x_final)
    autoencoder.compile(optimizer='adam', loss='mae')

    return autoencoder


def Surrogate(grid=(160, 160), nfreq=32, ntheta=24, dim=1):
    # Original Model
    input_bathy = Input(shape=(grid[0], grid[1], dim))
    input_spec = Input(shape=(nfreq, ntheta, dim))

    flat_spec = Flatten()(input_spec)
    flat_spec = Dense(nfreq * ntheta, activation='relu')(flat_spec)
    flat_spec = Dense(grid[0] * grid[1], activation='relu')(flat_spec)

    spec = Reshape((grid[0], grid[1], dim))(flat_spec)
    input_img_up = Concatenate()([spec, input_bathy])

    down_1 = MaxPooling2D((8, 8), padding='same')(input_img_up)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(down_1)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x1 = UpSampling2D((2, 2))(x1)

    down_2 = MaxPooling2D((4, 4), padding='same')(input_img_up)
    x2 = Concatenate()([x1, down_2])
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
    x2 = UpSampling2D((2, 2))(x2)

    down_3 = MaxPooling2D((2, 2), padding='same')(input_img_up)
    x3 = Concatenate()([x2, down_3])
    x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(x3)
    x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(x3)
    x3 = UpSampling2D((2, 2))(x3)

    x4 = Concatenate()([x3, input_img_up])
    x4 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
    x4 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)

    # Multi-scale model (Du et al., 2018)
    layer_1 = Conv2D(16, (5, 5), activation='relu', padding='same')(input_img_up)
    x1m = Conv2D(8, (5, 5), activation='relu', padding='same')(layer_1)
    x1m = Conv2D(8, (5, 5), activation='relu', padding='same')(x1m)

    layer_2 = Conv2D(16, (9, 9), activation='relu', padding='same')(input_img_up)
    x2m = Conv2D(8, (9, 9), activation='relu', padding='same')(layer_2)
    x2m = Conv2D(8, (9, 9), activation='relu', padding='same')(x2m)

    layer_3 = Conv2D(16, (13, 13), activation='relu', padding='same')(input_img_up)
    x3m = Conv2D(8, (13, 13), activation='relu', padding='same')(layer_3)
    x3m = Conv2D(8, (13, 13), activation='relu', padding='same')(x3m)

    x_add = Concatenate()([x1m, x2m, x3m, input_img_up])
    x4m = Conv2D(8, (7, 7), activation='relu', padding='same')(x_add)
    x4m = Conv2D(3, (5, 5), activation='relu', padding='same')(x4m)

    x_final = Concatenate()([x4, x4m])
    x_final = Conv2D(dim, (3, 3), padding='same')(x_final)
    autoencoder = Model([input_spec, input_bathy], x_final)
    autoencoder.compile(optimizer='adam', loss='mae')

    return autoencoder

