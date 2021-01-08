"""Class that load and processes daily data streams for prediction."""

import glob
import os
import pickle
import logging
import numpy as np

from scipy.signal import stft
from obspy import read, read_inventory

from local_config import ROOT_PATH


INVENTORY_FILE = os.path.join(ROOT_PATH, 'inputdata/NL_stations_2019.xml')
LOCAL_FORMAT = '%s/H[GH][12ENZ].D/NL*.%03d'
DEFAULT_WINDOW_SIZE = 20
DEFAULT_STRIDE = 10
TARGET_FREQ = 100


logger = logging.getLogger('dailystream.py')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter('%(asctime)s %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


class DailyStream:
    """Encapsulates a daily stream.

    Parameters
    ----------
    root_dir : `str`
        Root of the SDS data archive directory.
    station : `str`
        Station code.
    day : `int`
        Julian day of the stream.

    """

    _inventory = read_inventory(INVENTORY_FILE)

    def __init__(self, root_dir, station, day):
        self.station = station
        self.day = day
        self.stream = None

        filename_format = os.path.join(root_dir, LOCAL_FORMAT)
        file_list = glob.glob(filename_format % (station, day))
        if len(file_list) == 3:
            self.stream = read(file_list[0]) + read(file_list[1]) + read(file_list[2])
            self.stream.sort(['channel'])

            # Skip rotation error
            try:
                self.stream.rotate(method="->ZNE", inventory=self._inventory)
            except (NotImplementedError, Exception):
                logger.error('Error rotating the stream.')
                self.stream = None
                return

        else:
            logger.debug('Wrong number of files for station %s day %d', station, day)

    def samples(self, window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE,
                compute_frequencies=True):
        """Generate samples from the stream."""
        start = max([tr.stats.starttime for tr in self.stream])
        end = start + window_size

        while end <= min([tr.stats.endtime for tr in self.stream]):
            # Slice sample
            sample = self.stream.slice(start, end).copy()

            sample.split()
            if len(sample) != 3:
                raise ValueError('Sample does not have 3 traces.')

            # Store maximum amplitude
            stream_max = np.absolute(sample.max()).max()

            # Filter, resample, and normalize sample
            sample.detrend(type='linear')
            sample.filter('bandpass', freqmin=0.5, freqmax=22)
            sample.resample(TARGET_FREQ)
            sample.normalize(global_max=True)

            # TODO: compute frequency data
            freq_data = None
            if compute_frequencies:
                freq_data = []
                for trace in sample:
                    _, _, zxx = stft(trace.data, window='hanning', nperseg=120)
                    freq_data.append(np.abs(zxx))

            # Format the input for prediction
            x = np.array([sample[0].data, sample[1].data, sample[2].data])
            x = np.transpose(x)
            x = x.reshape(window_size * TARGET_FREQ, 3)

            stream_max = np.array(stream_max)

            yield x, stream_max, start, freq_data

            # Next start/end times
            start = start + stride
            end = start + window_size

    @classmethod
    def _get_coordinates(cls, station_code):
        search_result = cls._inventory.select(station=station_code)

        if len(search_result) != 1:
            raise KeyError('number of networks found != 1')
        if len(search_result[0]) != 1:
            raise KeyError('number of stations found != 1')

        station = search_result[0][0]

        lat = station.latitude
        lon = station.longitude
        return lat, lon

    def get_station_coordinates(self):
        """Return the coordinates for the associated station."""
        return self._get_coordinates(self.station)

    def load_samples(self, compute_frequencies=True):
        """Load and computes samples from the stream.

        Parameters
        ----------
        compute_frequencies : `bool`
            Whether or not to compute and save frequency data for each sample (default True).

        Returns
        -------
        input_data : `list`
            Contains the input to the predict method. The first element is a
            sequence of `np.array` objects with the normalized stream amplitude
            data, the second is a sequence of the maximum values before
            normalization. Is `None` if there's no data.
        start_times : `tuple` of `obspy.UTCDateTime`
            The start timestamps of each sample.
        freq_data : `tuple` or `None`
            If `compute_frequencies` is `True`, contains the frequency data of the samples.

        """
        # Process stream
        day_samples = [(sample_stream, stream_max, start_time, freq_data)
                       for sample_stream, stream_max, start_time, freq_data
                       in self.samples(compute_frequencies=compute_frequencies)]
        day_samples = [tup for tup in zip(*day_samples)]

        # If empty, exit
        if len(day_samples) == 0:
            logger.debug('No data')
            return None, None, None

        # Split the data
        input_data = day_samples[0:2]
        start_times = day_samples[2]

        freq_data = None
        if compute_frequencies:
            freq_data = day_samples[3]

        return input_data, start_times, freq_data

    def pickle_samples(self, pickle_dir, compute_frequencies=True):
        """Save the samples for prediction in pickle files.

        Parameters
        ----------
        pickle_dir : `str`
            Path of the directory that contains the pickle files.
        compute_frequencies : `bool`
            Whether or not to compute and save frequency data for each sample (default True).

        """
        input_data, start_times, freq_data = self.load_samples(compute_frequencies)

        if input_data is None:
            logger.debug('Not saving files')
            return

        # Create directory if necessary
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)

        # Save data
        sta_day = '{}_{:03d}'.format(self.station, self.day)
        input_data_filename = os.path.join(pickle_dir, sta_day + '_data.pkl')
        pickle.dump(input_data, open(input_data_filename, 'wb'))
        start_times_filename = os.path.join(pickle_dir, sta_day + '_times.pkl')
        pickle.dump(start_times, open(start_times_filename, 'wb'))

        if compute_frequencies:
            frequency_data_filename = os.path.join(pickle_dir, sta_day + '_freq.pkl')
            pickle.dump(freq_data, open(frequency_data_filename, 'wb'))

    @classmethod
    def unpickle_samples(cls, pickle_dir, station, day, load_frequencies=True):
        """Load the samples for prediction from pickle files.

        Parameters
        ----------
        pickle_dir : `str`
            Path of the directory that contains the pickle files.
        station : `str`
            Station code.
        day : `int`
            Julian day of the stream.
        load_frequencies : `bool`
            Whether or not to look for frequency data for each sample (default True).

        Returns
        -------
        loaded_data : `dict`
            A dictionary containing all the data loaded from the pickle files
            to be used for prediction. It contains the following fields:
            * amplitude_data : `list`
                Contains the amplitude input to the predict method. The first
                element is a sequence of `np.array` objects with the normalized
                stream amplitude data, the second is a sequence of the maximum
                values before normalization.
            * start_times : `tuple` of `obspy.UTCDateTime`
                The start timestamps of each sample.
            * latitude : `float`
                The station latitude.
            * longitude : `float`
                The station longitude.
            * frequency_data : `tuple` of `np.array`
                (optional) The frequency data of each sample to use in
                prediction.

        """
        sta_day = '{}_{:03d}'.format(station, day)
        amplitude_data_filename = os.path.join(pickle_dir, sta_day + '_data.pkl')
        start_times_filename = os.path.join(pickle_dir, sta_day + '_times.pkl')

        amplitude_data = pickle.load(open(amplitude_data_filename, 'rb'))
        start_times = pickle.load(open(start_times_filename, 'rb'))

        try:
            lat, lon = cls._get_coordinates(station)
        except KeyError:
            logger.debug('Could not get station coordinates.')
            lat = 0.0
            lon = 0.0

        loaded_data = {
            'amplitude_data': amplitude_data,
            'start_times': start_times,
            'latitude': lat,
            'longitude': lon
        }
        if load_frequencies:
            frequency_data_filename = os.path.join(pickle_dir, sta_day + '_freq.pkl')
            loaded_data['frequency_data'] = pickle.load(open(frequency_data_filename, 'rb'))

        return loaded_data

    @staticmethod
    def is_pickled(pickle_dir, station, day):
        """Check whether stream is preprocessed and saved in pickle files.

        Does not check whether frequency data is pickled.

        Parameters
        ----------
        pickle_dir : `str`
            Path of the directory that contains the pickle files.
        station : `str`
            Station code.
        day : `int`
            Julian day of the stream.

        Returns
        -------
        `bool`

        """
        sta_day = '{}_{:03d}'.format(station, day)
        amplitude_data_filename = os.path.join(pickle_dir, sta_day + '_data.pkl')
        start_times_filename = os.path.join(pickle_dir, sta_day + '_times.pkl')

        return os.path.exists(amplitude_data_filename) and os.path.exists(start_times_filename)


def get_stations(data_dir):
    """Return a list of the names of the immediate subdirectories of `data_dir`."""
    return sorted(next(os.walk(data_dir))[1])

    # return ['G101', 'G102', 'G103', 'G104', 'G111', 'G112', 'G114',
    # 'G121', 'G122', 'G123', 'G124', 'G131', 'G133', 'G134', 'G141',
    # 'G143', 'G144', 'G161', 'G162', 'G163', 'G164', 'G171', 'G173',
    # 'G181', 'G182', 'G184', 'G191', 'G193', 'G194', 'G402', 'G403',
    # 'G404', 'G411', 'G412', 'G413', 'G421', 'G422', 'G423', 'G424',
    # 'G431', 'G433', 'G434', 'G441', 'G443', 'G451', 'G452', 'G453',
    # 'G464', 'G471', 'G472', 'G473', 'G474', 'G482', 'G483', 'G484',
    # 'G492', 'G493', 'G601', 'G603', 'G604', 'G611', 'G613', 'G614',
    # 'G621', 'G622', 'G623', 'G624', 'G631', 'G632', 'G633', 'G634',
    # 'G642', 'G643', 'G654', 'G661', 'G662', 'G663']

    # return ['G492', 'G493', 'G601', 'G603', 'G604', 'G611', 'G613',
    # 'G614', 'G621', 'G622', 'G623', 'G624', 'G631', 'G632', 'G633',
    # 'G634', 'G642', 'G643', 'G654', 'G661', 'G662', 'G663']

    # return ['BSTD', 'BAPP', 'G493', 'G144', 'BWIN']
