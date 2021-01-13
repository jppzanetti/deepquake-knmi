import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.python.ops import math_ops


_DATA_STREAM_LENGTH = 2000


_FEATURES = {
    'window_size': tf.io.FixedLenFeature([], tf.int64),
    'n_traces': tf.io.FixedLenFeature([], tf.int64),
    'time_data': tf.io.FixedLenFeature([], tf.string),
    'stream_max': tf.io.FixedLenFeature([], tf.float32),
    'event_type': tf.io.FixedLenFeature([], tf.int64),
    'distance': tf.io.FixedLenFeature([], tf.float32),
    'magnitude': tf.io.FixedLenFeature([], tf.float32),
    'depth': tf.io.FixedLenFeature([], tf.float32),
    'azimuth': tf.io.FixedLenFeature([], tf.float32),
    'start_time': tf.io.FixedLenFeature([], tf.int64),
    'end_time': tf.io.FixedLenFeature([], tf.int64),
    'spec0': tf.io.FixedLenFeature([], tf.string),
    'spec1': tf.io.FixedLenFeature([], tf.string),
    'spec2': tf.io.FixedLenFeature([], tf.string)
}


def remove_surface_stations(files):
    #removing the G stations that are at ground level (0 last digit in the GXXX format)
    regexToMatchRemoval = ".*NL_G([0-9])([0-9])([0]).*tfrecords"
    r = re.compile(regexToMatchRemoval)
    newList = list(filter(r.match, files))

    # others = ".*NL_G([0-9])([0-9])([1-9]).*tfrecords"
    # r1 = re.compile(others)
    # newlist1 = list(filter(r1.match, files))
    # print(newlist1)
    # print("---------------------")
    non_ground_files = [x for x in files if x not in newList]
    # print(newlist)
    # print("---------------------")
    return(non_ground_files)


def _initialize_classes_bins():
    """Defines the classes for each feature.

    Returns
    -------
    `dict` (`str` -> `numpy.Array`)
        Maps the name of the feature to the bin boundaries for that feature.
    """
    # Distance in KM
    distance_range = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 20, 30, 50])

    # Magnitude in Richter
    max_magnitude = 5
    min_magnitude = 0
    step_magnitude = 0.2
    magnitude_range = np.arange(min_magnitude + step_magnitude,
                                max_magnitude + step_magnitude,
                                step_magnitude)

    # Depth in kilometers
    max_depth = 20
    min_depth = 0
    step_depth = 2
    depth_range = np.arange(min_depth, max_depth + step_depth, step_depth)

    # Azimuth in degrees
    max_azimuth = 180
    min_azimuth = 0
    step_azimuth = 15
    azimuth_range = np.arange(min_azimuth + step_azimuth,
                              max_azimuth + step_azimuth,
                              step_azimuth)

    # Classes of streams --- noise, seismic event, explosion, acoustic event
    detection_range = np.arange(0, 3, 1)

    return {'distance': distance_range,
            'magnitude': magnitude_range,
            'depth': depth_range,
            'azimuth': azimuth_range,
            'detection': detection_range}


CLASSES_BINS = _initialize_classes_bins()


def _cnn_layers_time(amp_input, freq_input, maxValStream, feature_name):
    if feature_name == 'detection':
        output = detection_NN_architecture_time(amp_input)
    if feature_name == 'magnitude':
        output = magnitude_NN_architecture(amp_input, maxValStream)
    if feature_name == 'azimuth':
        output = azimuth_NN_architecture(amp_input, maxValStream)

    return output


def _cnn_layers_freq(amp_input, freq_input, maxValStream, feature_name):
    if feature_name == 'detection':
        output = detection_NN_architecture_freq(freq_input)
    if feature_name == 'magnitude':
        output = magnitude_NN_architecture_freq(freq_input, maxValStream)
    if feature_name == 'azimuth':
        output = azimuth_NN_architecture_freq(freq_input, maxValStream)

    return output


def detection_NN_architecture_time(inputs):
    feature_name = "detection"
    x = layers.Conv1D(64, activation='relu', batch_input_shape=(None, _DATA_STREAM_LENGTH, 3),
                      data_format='channels_last', kernel_size=10, strides=4,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(inputs)
    x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    # #x = layers.Dropout(0.1)(x)
    # x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
    #                   kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    # #x = layers.Dropout(0.1)(x)
    # x = layers.Conv1D(32, activation='relu', kernel_size=5, strides=2,
    #                 kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    # x = layers.Dropout(0.25)(x)
    # x = layers.Conv1D(32, activation='elu', kernel_size=3, strides=2,
    #                 kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Flatten()(x)

    last = layers.Dense(64, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(x)
    output = layers.Dense(len(CLASSES_BINS[feature_name]),
                          activation='softmax', name=(feature_name + '_out'))(last)

    return output

def detection_NN_architecture_freq_and_time(amp_input, freq_input):
    feature_name = "detection"
    x = layers.Conv1D(64, activation='relu', batch_input_shape=(None, _DATA_STREAM_LENGTH, 3),
                      data_format='channels_last', kernel_size=10, strides=4,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(amp_input)
    x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Flatten()(x)

    # Spectrogram inputs
    y = layers.Conv2D(64, activation='relu', batch_input_shape=(None, (61, 35), 3),
                      data_format='channels_last', kernel_size=(2, 2), strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001))(freq_input)

    y = layers.Conv2D(64, activation='relu', kernel_size=(2, 2), strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), padding='valid')(y)

    y = layers.Conv2D(64, activation='relu', kernel_size=(2, 2), strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), padding='valid')(y)

    y = layers.Conv2D(64, activation='relu', kernel_size=(2, 2), strides=1,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), padding='valid')(y)
    y = layers.Conv2D(64, activation='relu', kernel_size=(2, 2), strides=1,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), padding='valid')(y)
    y = layers.Conv2D(64, activation='relu', kernel_size=(2, 2), strides=1,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), padding='valid')(y)

    y = layers.Flatten()(y)

    combined = layers.Concatenate(axis=1)([x, y])

    last = layers.Dense(64, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(l=0.0001))(combined)
    output = layers.Dense(len(CLASSES_BINS[feature_name]),
                          activation='softmax', name=(feature_name + '_out'))(last)

    return output


def detection_NN_architecture_freq(freq_input):
    feature_name = "detection"
    # Spectrogram inputs
    y = layers.Conv2D(64, activation='relu', batch_input_shape=(None, (61, 35), 3),
                      data_format='channels_last', kernel_size=(2, 2), strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001))(freq_input)

    y = layers.Conv2D(64, activation='relu', kernel_size=(2, 2), strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), padding='valid')(y)

    y = layers.Conv2D(64, activation='relu', kernel_size=(2, 2), strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), padding='valid')(y)

    y = layers.Conv2D(64, activation='relu', kernel_size=(2, 2), strides=1,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), padding='valid')(y)
    y = layers.Conv2D(64, activation='relu', kernel_size=(2, 2), strides=1,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), padding='valid')(y)
    y = layers.Conv2D(64, activation='relu', kernel_size=(2, 2), strides=1,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), padding='valid')(y)

    y = layers.Flatten()(y)

    output = layers.Dense(len(CLASSES_BINS[feature_name]),
                          activation='softmax', name=(feature_name + '_out'))(y)

    return output


def magnitude_NN_architecture(inputs, maxValStream):
    feature_name = "magnitude"
    x = layers.Conv1D(512, activation='relu', batch_input_shape=(None, _DATA_STREAM_LENGTH, 3),
                      data_format='channels_last', kernel_size=6, strides=4,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(inputs)
    x = layers.Conv1D(350, activation='relu', kernel_size=5, strides=1,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(256, activation='relu', kernel_size=5, strides=1,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(196, activation='relu', kernel_size=5, strides=1,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(128, activation='relu', kernel_size=5, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=5, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=3, strides=1,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=3, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Flatten()(x)
    maxValStream = tf.expand_dims(maxValStream, 1)
    x = layers.Concatenate(axis=1)([x, maxValStream])

    last = layers.Dense(64, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(x)
    output = layers.Dense(len(CLASSES_BINS[feature_name]),
                          activation='softmax', name=(feature_name + '_out'))(last)

    return output


def azimuth_NN_architecture(inputs, maxValStream):
    feature_name = "azimuth"
    x = layers.Conv1D(128, activation='relu', batch_input_shape=(None, _DATA_STREAM_LENGTH, 3),
                      data_format='channels_last', kernel_size=80, strides=3,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(inputs)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=20, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=20, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=20, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Dropout(0.25)(x)
   # x = layers.Conv1D(64, activation='relu', kernel_size=3, strides=2,
   #                   kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
   # x = layers.Dropout(0.1)(x)
   # x = layers.Conv1D(64, activation='relu', kernel_size=3, strides=2,
   #                   kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
   # x = layers.Dropout(0.1)(x)
   # x = layers.Conv1D(64, activation='relu', kernel_size=3, strides=2,
   #                    kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
   # x = layers.Dropout(0.1)(x)
   # x = layers.Conv1D(64, activation='relu', kernel_size=3, strides=2,
   #                  kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    # x = layers.Dropout(0.25)(x)
    # x = layers.Conv1D(32, activation='elu', kernel_size=3, strides=2,
    #                 kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Flatten()(x)

    #maxValStream = tf.expand_dims(maxValStream, 1)
    #x = layers.Concatenate(axis=1)([x, maxValStream])

    last = layers.Dense(64, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(x)
    output = layers.Dense(len(CLASSES_BINS[feature_name]),
                          activation='softmax', name=(feature_name + '_out'))(last)

    return output


def load_model(feature_name, use_frequency=False):
    """Load and compile the CNN model for the given feature.

    Parameters
    ----------
    feature_name : `str`
        Name of the feature to be loaded (`"detection"`, `"azimuth"`,
        `"depth"`, `"distance"`, `"magnitude"`).
    use_frequency : `bool`
        Load model using frequency data instead of the stream timeseries. (default `False`)

    Returns
    -------
    `Model`
        A TensorFlow model of the neural network.

    """
    model_input_amp = layers.Input(shape=(_DATA_STREAM_LENGTH, 3))
    model_input_freq = layers.Input(shape=(61, 35, 3))
    model_input_max = layers.Input(shape=())

    if use_frequency:
        model_output = _cnn_layers_freq(model_input_amp, model_input_freq, model_input_max,
                                        feature_name)
        model = Model(inputs=[model_input_amp, model_input_freq, model_input_max],
                      outputs=model_output)
    else:
        model_output = _cnn_layers_time(model_input_amp, model_input_freq, model_input_max,
                                        feature_name)
        model = Model(inputs=[model_input_amp, model_input_freq, model_input_max],
                      outputs=model_output)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def _extract_fn(data_record, feature_name):
    """Extract the data from a single TFRecord Tensor.

    Parameters
    ----------
    data_record : `tf.Tensor`
        The Tensor object from the sample's TFRecord.
    feature_name : `str`
        The name of the feature to be loaded.

    Returns
    -------
    data : `tf.Tensor`
        Waveform data of the sample.
    maxValStream : `tf.Tensor`
        Maximum amplitude before normalization.
    out_label : `tf.Tensor (?)`
        Target class for the given feature in this sample.

    """
    # Parse record
    all_data = _FEATURES
    sampleData = tf.io.parse_single_example(data_record, all_data)

    # Convert and reshape stream data
    time_data = tf.io.decode_raw(sampleData['time_data'], tf.float32)
    time_data = tf.slice(time_data, [0], [3 * _DATA_STREAM_LENGTH])
    time_data.set_shape([3 * _DATA_STREAM_LENGTH])
    time_data = tf.reshape(time_data, [3, _DATA_STREAM_LENGTH])
    time_data = tf.transpose(time_data, [1, 0])

    # Convert and reshape spectrogram data
    amp1 = tf.io.decode_raw(sampleData['spec0'], tf.float64)
    amp2 = tf.io.decode_raw(sampleData['spec1'], tf.float64)
    amp3 = tf.io.decode_raw(sampleData['spec2'], tf.float64)
    amp1.set_shape([61 * 35])
    amp2.set_shape([61 * 35])
    amp3.set_shape([61 * 35])
    freq_data = tf.concat([amp1, amp2, amp3], 0)
    freq_data = tf.reshape(freq_data, [3, 61, 35])
    freq_data = tf.transpose(freq_data, [1, 2, 0])

    # Read stream max amplitude
    maxValStream = sampleData['stream_max']

    # Dictionary to hold output classes
    classes = {}

    # Read event type
    classes['detection'] = math_ops._bucketize(tf.math.subtract(sampleData['event_type'], 1),
                                               boundaries=list(CLASSES_BINS['detection']))

    # Compute azimuth class
    azim_shape = tf.shape(sampleData['azimuth'])
    azim_flat = tf.reshape(sampleData['azimuth'], [-1])
    azim_180_flat = tf.map_fn(lambda x: x % 180, azim_flat)
    azim_180 = tf.reshape(azim_180_flat, azim_shape)
    classes['azimuth'] = math_ops._bucketize(azim_180, boundaries=list(CLASSES_BINS['azimuth']))

    # For other features, simply read value and bucketize it
    for prop_name in ['depth', 'magnitude', 'distance']:
        classes[prop_name] = math_ops._bucketize(sampleData[prop_name],
                                                 boundaries=list(CLASSES_BINS[prop_name]))

    # Get desired feature
    if feature_name in classes:
        out_label = classes[feature_name]
    else:
        raise ValueError('Invalid feature name.')

    return time_data, freq_data, maxValStream, out_label


def load_dataset(filenames, feature_name):
    """Load the dataset contained in the given filenames.

    Parameters
    ----------
    filenames : `list` of `str`
        List of the .tfrecords files to be included.
    feature_name : `str`
        Name of the feature to be loaded (`"detection"`, `"azimuth"`,
        `"depth"`, `"distance"`, `"magnitude"`).

    Returns
    -------
    `TFRecordDataset`
        A TensorFlow container of the whole dataset.

    """
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda x: _extract_fn(x, feature_name))

    time_inputs = dataset.map(lambda t, s, m, y: t)
    freq_inputs = dataset.map(lambda t, s, m, y: s)
    maxval = dataset.map(lambda t, s, m, y: m)
    targets = dataset.map(lambda t, s, m, y: y)

    full_dataset = tf.data.Dataset.zip((tf.data.Dataset.zip((time_inputs, freq_inputs, maxval)),
                                        targets))

    return full_dataset
