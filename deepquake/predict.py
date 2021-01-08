import os
import json
import argparse
import time
import logging
import datetime

import numpy as np
import tensorflow as tf
from scipy.signal import stft

from models import load_model
from dailystream import DailyStream, get_stations

EXTRA_FEATURES = []

# Workaround to fix logging, might break tf logging
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

logger = logging.getLogger('predict.py')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


def process_single_day(data_dir, pickle_dir, models, station, day, output_mode,
                       use_frequencies=None):
    """Execute the prediction on a single day of a single station.

    Parameters
    ----------
    data_dir : `str`
        Path to the directory containing the data files.
    pickle_dir : `str`
        Path to the directory containing the pickle files.
    models : `dict` (`str` -> `tensorflow.keras.models.Model`)
        Trained TensorFlow models predicting each feature in `['detection']` + `EXTRA_FEATURES`.
    station : `str`
        Station code.
    day : `int`
        Day of the year (1-365/366).
    output_mode : `str`
        How to output results --- `'stdout'` or `'mongo'`.
    use_frequencies : `str`
        `None` (default), `'load'`, `'compute'`

    """
    # Try to load pickle
    if DailyStream.is_pickled(pickle_dir, station, day):
        step_start = time.perf_counter()
        logger.debug('Loading pickle files')

        # Are we loading frequency data from pickles?
        load_frequencies = False
        if use_frequencies == 'load':
            load_frequencies = True

        sample_info = DailyStream.unpickle_samples(pickle_dir, station, day,
                                                   load_frequencies=load_frequencies)
        input_data = sample_info['amplitude_data']
        start_times = sample_info['start_times']
        latitude = sample_info['latitude']
        longitude = sample_info['longitude']

        if use_frequencies == 'load':
            input_data = sample_info['frequency_data']

        logger.debug('Number of samples: %d', len(sample_info['start_times']))
        logger.debug('Step: %.4f', time.perf_counter() - step_start)

    # Day is not preprocessed
    else:
        step_start = time.perf_counter()
        logger.debug('Loading mseed files')

        stream = DailyStream(data_dir, station, day)
        if stream.stream is None:
            logger.debug('No data')
            return

        input_data, start_times, _ = stream.load_samples(compute_frequencies=False)
        if input_data is None:
            logger.debug('No data')
            return

        latitude, longitude = stream.get_station_coordinates()

        logger.debug('Number of samples: %d', len(start_times))
        logger.debug('Step: %.4f', time.perf_counter() - step_start)

    # Compute frequencies if needed
    if use_frequencies == 'compute':
        step_start = time.perf_counter()
        logger.debug('Computing frequencies')

        frequency_data = []
        for sample_amplitudes in input_data[0]:
            sample_freq = []

            # Every channel is a column in the sample
            for trace in sample_amplitudes.T:
                _, _, zxx = stft(trace.data, window='hanning', nperseg=120)  # TODO I need to use nperseg=120 to get 61x35 results, why?
                sample_freq.append(np.abs(zxx))

            # Stack the arrays together
            freq_array = np.stack(sample_freq, axis=2)
            frequency_data.append(freq_array)

        input_data[0] = frequency_data
        logger.debug('Step: %.4f', time.perf_counter() - step_start)

    step_start = time.perf_counter()
    logger.debug('Going to prediction')

    # Classify the sample
    prediction = models['detection'].predict(input_data)
    # logger.debug(prediction)

    logger.debug('Step: %.4f', time.perf_counter() - step_start)
    step_start = time.perf_counter()
    logger.debug('Reading prediction results')

    detections = 0

    pred_classes = [np.argmax(x) for x in prediction]
    logger.debug('0: %d 1: %d 2: %d',
                 pred_classes.count(0),
                 pred_classes.count(1),
                 pred_classes.count(2))

    for i, p in enumerate(prediction):
        event_type = np.argmax(p)
        if event_type > 0:
            detections += 1

            prediction_time = datetime.datetime.utcfromtimestamp(start_times[i].timestamp)

            # Save classes in a dict/JSON
            result = {
                'type': int(event_type),
                'station': station,
                'station_lat': latitude,
                'station_lon': longitude,
                'start_time': prediction_time,
                'jday': day,
                'year': prediction_time.year,
                'detection_activation': float(p[event_type]),
                'processing_timestamp': datetime.datetime.now()
            }

            # Predict event's features
            for f in EXTRA_FEATURES:
                # TODO: fix these reshapes, they're not valid for frequency
                result[f] = int(np.argmax(models[f].predict([input_data[0][i].reshape(1, 2000, 3),
                                                             input_data[1][i].reshape(-1)])))

            if output_mode == 'mongo':
                logger.debug(result)
                mongo_session.save_result(result)
            else:
                result['start_time'] = str(start_times[i])
                result['processing_timestamp'] = result['processing_timestamp'].isoformat()
                print(json.dumps(result))

    logger.info('Detections: %d', detections)
    logger.debug('Step: %.4f', time.perf_counter() - step_start)


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', choices=['stdout', 'mongo'], required=True,
                        help='Where to output results.')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the directory containing the data files.')
    parser.add_argument('--pickle_dir', type=str,
                        help='Path to the directory containing the pickle files.')
    parser.add_argument('--start_day', type=int, default=1,
                        help='First day of prediction.')
    parser.add_argument('--end_day', type=int, default=206,
                        help='Last day of prediction.')
    parser.add_argument('--day_list', nargs='+', type=int,
                        help='A list of days to process (overrides --start_day and --end_day).')
    parser.add_argument('--frequency', action='store_true',
                        help='Predict on spectrograms instead of the streams.')
    parser.add_argument('--mongo_collection', type=str,
                        help='Name for the MongoDB collection to store the predictions.')
    args = parser.parse_args()

    # Tensorflow version
    logger.debug(tf.__version__)

    if args.output == 'mongo':
        from mongo_logger import mongo_session
        mongo_session.detection_collection = args.mongo_collection

    # Load trained models
    models = {}
    for f in ['detection'] + EXTRA_FEATURES:
        model = load_model(f, use_frequency=args.frequency)
        if args.frequency:
            model.load_weights(os.path.join('weights', f + '_freq_saved_wt.h5'))
        else:
            model.load_weights(os.path.join('weights', f + '_time_saved_wt.h5'))
        models[f] = model
        models[f].summary(print_fn=logger.debug)

    station_list = get_stations(args.data_dir)
    logger.debug(station_list)

    # Determine days to run the prediction
    days_to_predict = range(args.start_day, args.end_day + 1)
    if args.day_list:
        days_to_predict = args.day_list

    # Iterate over stations and days
    for station in station_list:
        for day in days_to_predict:
            logger.info('%s %d', station, day)
            step_start = time.perf_counter()
            if args.frequency:
                process_single_day(args.data_dir, args.pickle_dir,
                                   models, station, day, args.output,
                                   use_frequencies='compute')
            else:
                process_single_day(args.data_dir, args.pickle_dir,
                                   models, station, day, args.output)
            logger.debug('Day: %.4f', time.perf_counter() - step_start)
