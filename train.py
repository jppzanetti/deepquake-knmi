import random
import glob
import argparse
import datetime

import tensorflow as tf
from sklearn.metrics import classification_report

from models import load_model, load_dataset, remove_surface_stations
import local_config
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


if __name__ == '__main__':
    # Seed for shuffling files
    random.seed(1)

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature',
                        choices=['detection', 'distance', 'magnitude', 'azimuth'],
                        required=True,
                        help='Which feature to train.')
    parser.add_argument('--num_epochs', type=int, required=True,
                        help='Number of epochs to train.')
    parser.add_argument('--remove_surface_stations', action='store_true',
                        help='Remove all G??0 stations from input.')
    args = parser.parse_args()
    epochsTrain = args.num_epochs

    streamsPathEvent = local_config.PATH_TRAIN_EVENT
    evalPathEvent = local_config.PATH_EVAL_EVENT

    files = (glob.glob(streamsPathEvent + "/*.tfrecords"))
    eval_files = (glob.glob(evalPathEvent + "/*.tfrecords"))

    if args.feature == 'detection':
        streamsPathNoise = local_config.PATH_TRAIN_NOISE
        evalPathNoise = local_config.PATH_EVAL_NOISE
        files = files + glob.glob(streamsPathNoise + "/*.tfrecords")
        eval_files = eval_files + glob.glob(evalPathNoise + "/*.tfrecords")

    # Remove all samples from G??0 stations
    if args.remove_surface_stations:
        files = remove_surface_stations(files)
        print("# Files for training to be used "+str(len(files)))
        eval_files = remove_surface_stations(eval_files)
        print("# Files for validation to be used "+str(len(eval_files)))

    # Creating the NN model
    model = load_model(args.feature, use_frequency=True)
    print(model.summary())

    # Batch size
    bs = 128
    n_files = len(files)
    random.shuffle(files)
    train_data = load_dataset(files, args.feature)
    random.shuffle(eval_files)
    eval_data = load_dataset(eval_files, args.feature)

    # Note: be careful with the memory used by the cache()
    train_data = train_data.cache().shuffle(512).repeat(epochsTrain + 10).batch(bs)
    eval_data = eval_data.cache().shuffle(512).repeat(epochsTrain + 10).batch(bs)

    # log_dir = '~/tmp/' + args.feature + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
    #                                                       write_graph=True, write_images=True)

    stopping_callbacks = [
        EarlyStopping(patience=10, monitor='val_accuracy', min_delta=0, mode='max'),
        ModelCheckpoint(args.feature + '_saved_wt_best-MODEL.h5',
                        monitor='val_accuracy', save_best_only=True, save_weights_only=True,
                        verbose=1)
    ]

    steps = n_files // bs + 1
    val_steps = len(eval_files) // bs + 1
    model.fit(train_data,
              epochs=epochsTrain, steps_per_epoch=steps,
              verbose=1, callbacks=stopping_callbacks,
              validation_data=eval_data, validation_steps=val_steps)
