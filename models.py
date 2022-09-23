import tensorflow as tf
from tensorflow.keras import layers

def make_generator_model(out_shape, extra_layers=1):
    n_upsample = 3 + extra_layers

    outx = out_shape[0]
    outy = out_shape[1]
    model = tf.keras.Sequential()
    model.add(layers.Dense(int(outx / 2 ** (n_upsample - 1)) * int(outy / 2 ** (n_upsample - 1)) * 256, use_bias=False,
                           input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((int(outx / 2 ** (n_upsample - 1)), int(outy / 2 ** (n_upsample - 1)), 256)))
    assert model.output_shape == (
    None, int(outx / 2 ** (n_upsample - 1)), int(outy / 2 ** (n_upsample - 1)), 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, int(outx / 2 ** (n_upsample - 1)), int(outy / 2 ** (n_upsample - 1)), 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, int(outx / 2 ** (n_upsample - 2)), int(outy / 2 ** (n_upsample - 2)), 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    if extra_layers > 0:
        model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, int(outx / 2 ** (n_upsample - 3)), int(outy / 2 ** (n_upsample - 3)), 32)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

    if extra_layers > 1:
        model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, int(outx / 2 ** (n_upsample - 4)), int(outy / 2 ** (n_upsample - 4)), 16)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, outx, outy, 1)

    return model


def make_discriminator_model(in_shape):
    inx = in_shape[0]
    iny = in_shape[1]
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[inx, iny, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model