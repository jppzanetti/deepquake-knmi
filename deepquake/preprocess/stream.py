import obspy.geodetics.base as geo
from obspy import read
from obspy.taup import TauPyModel
from obspy.clients.fdsn.header import FDSNNoDataException, FDSNException
from http.client import IncompleteRead

import os
import numpy as np
import tensorflow as tf
import logging

from random import random

# Workaround to fix logging, might break tf logging
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False


SAMPLE_RATE = 100
WINDOW_LENGTH = 20
TAU_PY_MODEL_FILEPATH = 'taup_model/groningen.npz'
_VELOCITY_MODEL = TauPyModel(model=TAU_PY_MODEL_FILEPATH)
SDS_ROOT = '/nobackup/users/pereiraz/event_waveforms/deepquake/'

# Set up logging
logger = logging.getLogger('stream.py')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


def _fetch_fdsn(client, network_code, station_code, start_time, end_time):
    """Fetches a waveform stream through an FDSN client.

    Parameters
    ----------
    client : `obspy.clients.fdsn.Client`
        A client pointing to the appropriate FDSN web service.
    network_code : `str`
        Network code.
    station_code : `str`
        Station code.
    start_time : `obspy.UTCDateTime`
        The start time of the stream.
    end_time : `obspy.UTCDateTime`
        The end time of the stream.

    Returns
    -------
    `obspy.Stream` or `None`
    """
    try:
        source_stream = client.get_waveforms(network_code, station_code, '*', '*',
                                             start_time, end_time,
                                             attach_response=True)
        return source_stream

    except FDSNNoDataException:
        logger.error('No data available')
        return None
    except IncompleteRead:
        logger.error('Incomplete read')
        return None
    except FDSNException:
        logger.error('Another FDSN error')
        return None


def _fetch_local_sds(sds_root, network_code, station, start_time, end_time):
    """Fetches a waveform stream from a local SDS-like archive.

    Parameters
    ----------
    sds_root : `str`
        Path to the root directory of the SDS archive.
    network_code : `str`
        Network code.
    station : `obspy.Station`
        Station descriptor.
    start_time : `obspy.UTCDateTime`
        The start time of the stream.
    end_time : `obspy.UTCDateTime`
        The end time of the stream.

    Returns
    -------
    `obspy.Stream` or `None`
    """

    source_stream = None

    for channel in station:
        if 'LH' in channel.code:
            continue
        sds_path = ('{year}/{network}/{station}/{channel}.D/'
                    '{network}.{station}..{channel}.D.{year}.{julian_day}').format(
                        year=start_time.strftime('%Y'),
                        network=network_code,
                        station=station.code,
                        channel=channel.code,
                        julian_day=start_time.strftime('%j')
                    )
        if os.path.isfile(os.path.join(sds_root, sds_path)):
            channel_stream = read(os.path.join(sds_root, sds_path), format='MSEED',
                                  starttime=start_time, endtime=end_time)
            if source_stream is None:
                source_stream = channel_stream.copy()
            else:
                source_stream += channel_stream

    return source_stream


class Stream:
    """Base class for streams.

    Parameters
    ----------
    network : `str`
        Network code.
    station : `str`
        Station code.
    start_time : `obspy.UTCDateTime`
        The start time of the stream.
    source_raw_stream : `obspy.Stream`
        The raw stream data.
    inventory : `obspy.Inventory`
        Inventory containing station metadata.

    Attributes
    ----------
    network : `str`
        Network code.
    station : `str`
        Station code.
    start_time : `obspy.UTCDateTime`
        The start time of the stream.
    end_time : `obspy.UTCDateTime`
        The end time of the stream.
    raw_stream : `obspy.Stream`
        The raw stream data.
    filtered_stream : `obspy.Stream`
        The filtered stream data.
    stream_max : `float`
        The maximum amplitude of the raw stream.

    Raises
    ------
    ValueError
        Raised if stream is invalid.
    """

    def __init__(self, network, station, start_time, source_raw_stream, inventory):
        self.network = network
        self.station = station

        self.start_time = start_time
        self.end_time = self._end_time(start_time)

        # Validate raw stream
        self.raw_stream = source_raw_stream.copy()
        self.raw_stream.trim(starttime=self.start_time, endtime=self.end_time)
        if not self.is_valid(self.raw_stream):
            raise ValueError('Invalid stream')

        # Compute stream_max
        try:
            self.stream_max = np.absolute(self.raw_stream.max()).max()
        except (ValueError, AttributeError) as e:
            logger.error(self.raw_stream)
            logger.exception(e)
            self.stream_max = 0.0

        # Preprocess raw stream
        self.raw_preprocessing(inventory)

        # Filter stream
        self.filtered_stream = self.filter_stream()

    @staticmethod
    def _end_time(start_time):
        """Computes the end time for a sample with the given start time."""
        return start_time + WINDOW_LENGTH

    @classmethod
    def from_fdsn(cls, client, network, station, start_time, inventory):
        """Initializes a Stream object and fetches data from FDSN for it.

        Parameters
        ----------
        client : `obspy.clients.fdsn.Client`
            A client pointing to the appropriate FDSN web service.
        network : `str`
            Network code.
        station : `str`
            Station code.
        start_time : `obspy.UTCDateTime`
            The start time of the stream.
        inventory : `obspy.Inventory`
            Inventory containing station metadata.

        Returns
        -------
        `Stream`
        """

        # Fetches stream from FDSN
        end_time = cls._end_time(start_time)
        raw_stream = _fetch_fdsn(client, network, station, start_time, end_time)
        if raw_stream is None:
            return None

        stream = cls(network, station, start_time, raw_stream, inventory)
        return stream

    @classmethod
    def from_local_sds(cls, network, station, start_time, inventory):
        """Initializes a Stream object and fetches data from local storage for it.

        Parameters
        ----------
        network : `str`
            Network code.
        station : `str`
            Station code.
        start_time : `obspy.UTCDateTime`
            The start time of the stream.
        inventory : `obspy.Inventory`
            Inventory containing station metadata.

        Returns
        -------
        `Stream`
        """

        # Fetches stream from disk
        end_time = cls._end_time(start_time)
        raw_stream = _fetch_local_sds(SDS_ROOT, network, station, start_time, end_time)
        if raw_stream is None:
            return None

        stream = cls(network, station, start_time, raw_stream, inventory)
        return stream

    def is_valid(self, stream):
        """Checks integrity of stream."""

        # Null stream
        if stream is None:
            logger.debug('Null stream')
            return False

        # Number of traces
        if (len(stream) != 3
                or len({trace.stats.channel for trace in stream}) != 3):
            # Try to merge traces
            logger.debug('Merging stream')
            logger.debug('Before merging: %s', stream)
            stream.merge(method=1)
            logger.debug('After merging: %s', stream)

        # Test again
        if (len(stream) != 3
                or len({trace.stats.channel for trace in stream}) != 3):
            logger.debug('Wrong number of traces/channels in stream')
            return False

        # Time coverage
        for trace in stream:
            if len(trace) < 1:
                logger.debug('Empty trace')
                return False
            if (trace.stats.starttime > self.start_time + 1
                    or trace.stats.endtime < self.end_time - 1):
                logger.debug('Stream does not cover time window')
                logger.debug('Stream: %s --- %s Window: %s --- %s',
                             trace.stats.starttime.isoformat(), trace.stats.endtime.isoformat(),
                             self.start_time.isoformat(), self.end_time.isoformat())
                return False

        return True

    def raw_preprocessing(self, inventory):
        """Rotates, normalizes, and resamples the stream. Alters `self.raw_stream` in-place."""
        # Rotation
        self.raw_stream.rotate(method="->ZNE", inventory=inventory)
        self.raw_stream.sort(['channel'])

        # Resampling
        for trace in self.raw_stream:
            trace.resample(SAMPLE_RATE)

        # Normalization
        self.raw_stream.normalize(global_max=True)

    def filter_stream(self):
        """Makes a copy of the raw stream and applies filters to it."""
        filtered_stream = self.raw_stream.copy()
        filtered_stream.detrend(type='linear')
        filtered_stream.filter('bandpass', freqmin=0.5, freqmax=22)

        # Re-normalize stream
        filtered_stream.normalize(global_max=True)

        return filtered_stream

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _save_tfrecords_file(stream, stream_max, filepath,
                             event_type=0, distance=-1, magnitude=-1, depth=-1, azimuth=-1):
        """Save one sample stream in a .tfrecords file.

        Parameters
        ----------
        stream : `obspy.Stream`
            The raw stream data.
        stream_max : `float`
            The maximum amplitude of the raw stream.
        filepath : `str`
            Path and file name for the output file.
        """

        # Metadata
        n_traces = len(stream)
        n_samples = len(stream[0].data)
        start_time = np.int64(stream[0].stats.starttime.timestamp)
        end_time = np.int64(stream[0].stats.endtime.timestamp)

        # Format stream data
        data = np.zeros((n_traces, n_samples), dtype=np.float32)
        for i in range(n_traces):
            data[i, :] = stream[i].data[...]

        feature = {
            'window_size': Stream._int64_feature(n_samples),
            'n_traces': Stream._int64_feature(n_traces),
            'data': Stream._bytes_feature(data.tobytes()),
            'stream_max': Stream._float_feature(stream_max),
            'event_type': Stream._int64_feature(event_type),
            'distance': Stream._float_feature(distance),
            'magnitude': Stream._float_feature(magnitude),
            'depth': Stream._float_feature(depth),
            'azimuth': Stream._float_feature(azimuth),
            'start_time': Stream._int64_feature(start_time),
            'end_time': Stream._int64_feature(end_time)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        tfrecord_writer = tf.io.TFRecordWriter(filepath)
        tfrecord_writer.write(example.SerializeToString())

    def save_tfrecords(self, base_dir, relative_path, save_raw=False):
        """Save .tfrecords files relative to the `filtered_stream` (and
        optionally also the `raw_stream`) to a path given by
        `base_dir` + `'filter/'` or `'raw/'` + `relative_path`.
        """

        filtered_path = os.path.join(base_dir, 'filter', relative_path)
        self._save_tfrecords_file(self.filtered_stream, self.stream_max, filtered_path)
        if save_raw:
            raw_path = os.path.join(base_dir, 'raw', relative_path)
            self._save_tfrecords_file(self.raw_stream, self.stream_max, raw_path)


class EventStream(Stream):
    """Class for event streams.

    Parameters
    ----------
    client : `obspy.clients.fdsn.Client`
        A client pointing to the appropriate FDSN web service.
    event : `Event`
        The object for the event to look up.
    network_code : `str`
        Network code.
    station : `obspy.Station`
        The station object.
    source_raw_stream : `obspy.Stream`
        The raw stream data.
    inventory : `obspy.Inventory`
        Inventory containing station metadata.

    Attributes
    ----------
    station : `obspy.Station`
        The station object.
    event : `Event`
        The event information.
    magnitude : `float`
        Event magnitude.
    distance : `float`
        Distance between station and event epicenter in km.
    depth : `float`
        Event depth in km.
    eta : `obspy.UTCDateTime`
        Estimated time of arrival of the P-wave at the station.
    """

    def __init__(self, event, network_code, station, source_raw_stream, inventory):
        self.station = station
        self.event = event

        # Compute event features
        dist_meters, self.azimuth, _ = geo.gps2dist_azimuth(station.latitude,
                                                            station.longitude,
                                                            event.latitude,
                                                            event.longitude,
                                                            a=geo.WGS84_A, f=geo.WGS84_F)
        self.distance = dist_meters / 1000.0
        self.magnitude = event.magnitude
        self.depth = event.depth

        # Compute random start of window
        self.eta = self.estimate_arrival_time()
        window_offset = 1.5 + random() * 7
        start_time = self.eta - window_offset

        super().__init__(network_code, station.code, start_time, source_raw_stream, inventory)

        # Re-overwrite self.station with a Station instead of a string
        self.station = station

    @staticmethod
    def _compute_padding(event):
        """Computes start and end times with enough padding around the event.

        Parameters
        ----------
        event : `Event`

        Returns
        -------
        start_time : `obspy.UTCDateTime`
            Start time to fetch data.
        end_time : `obspy.UTCDateTime`
            End time to fetch data.
        """
        padding = 20.0
        max_travel_time = 30.0
        start_time = event.time - padding
        end_time = event.time + max_travel_time + WINDOW_LENGTH + padding
        return start_time, end_time

    @classmethod
    def augmented_set(cls, event, network_code, station, source_stream, inventory, num_samples=1):
        """Returns a list of samples from a given stream, each with random
        windows and noise (except for the first element, which has no
        noise added to it).

        Parameters
        ----------
        event : `Event`
            The object for the event to look up.
        network_code : `str`
            Network code.
        station : `obspy.Station`
            The station object.
        source_stream : `obspy.Stream`
            The source waveform data stream.
        inventory : `obspy.Inventory`
            Inventory containing station metadata.
        num_samples : `int`
            Number of samples (default 1).

        Returns
        -------
        `list` of `EventStream`
        """

        # Generate list of streams
        sample_list = []
        for _ in range(num_samples):
            try:
                stream = cls(event, network_code, station, source_stream, inventory)
                sample_list.append(stream)
                logger.debug(stream.raw_stream)
            except Exception as e:
                logger.exception(e)
                return None

        # Add noise
        for stream in sample_list[1:]:
            stream.add_noise()

        return sample_list

    @classmethod
    def from_fdsn(cls, client, event, network_code, station, inventory,
                  num_samples=1, mseed_path=None, save_mseed=False):
        """Returns a list of samples from the same event, each with random
        windows and noise (except for the first element, which has no
        noise added to it). Fetches data through an FDSN client.

        Parameters
        ----------
        client : `obspy.clients.fdsn.Client`
            A client pointing to the appropriate FDSN web service.
        event : `Event`
            The object for the event to look up.
        network_code : `str`
            Network code.
        station : `obspy.Station`
            The station object.
        inventory : `obspy.Inventory`
            Inventory containing station metadata.
        num_samples : `int`
            Number of samples (default 1).
        mseed_path : `str`
            Where to look for/save a miniSEED backup of the stream.
        save_mseed : `bool`
            Whether to save a miniSEED backup of the stream.

        Returns
        -------
        `list` of `EventStream`
        """

        if num_samples < 1:
            return None

        # Compute enough padding before and after event
        start_time, end_time = cls._compute_padding(event)

        # Check if stream is saved locally
        if os.path.isfile(mseed_path):
            source_stream = read(mseed_path, format='MSEED')
        else:
            source_stream = _fetch_fdsn(client, network_code, station.code,
                                        start_time, end_time)
        logger.debug(source_stream)

        # TODO: better validate this stream
        if source_stream is not None and len(source_stream) < 3:
            logger.debug('len(stream) < 3')
            source_stream = None

        # No data found
        if source_stream is None:
            return None

        # Save miniSEED file
        if save_mseed:
            source_stream.write(mseed_path, format='MSEED')

        # Generate streams
        sample_list = cls.augmented_set(event, network_code, station, source_stream, inventory,
                                        num_samples)
        return sample_list

    @classmethod
    def from_local_sds(cls, event, network_code, station, inventory, num_samples=1):
        """Returns a list of samples from the same event, each with random
        windows and noise (except for the first element, which has no
        noise added to it). Fetches data from a local SDS-like archive.

        Parameters
        ----------
        event : `Event`
            The object for the event to look up.
        network_code : `str`
            Network code.
        station : `obspy.Station`
            The station object.
        inventory : `obspy.Inventory`
            Inventory containing station metadata.
        num_samples : `int`
            Number of samples (default 1).

        Returns
        -------
        `list` of `EventStream`
        """

        if num_samples < 1:
            return None

        # Compute enough padding before and after event
        start_time, end_time = cls._compute_padding(event)

        # Get stream from local storage
        source_stream = _fetch_local_sds(SDS_ROOT, network_code, station,
                                         start_time, end_time)
        logger.debug(source_stream)

        # TODO: better validate this stream
        if source_stream is not None and len(source_stream) < 3:
            logger.debug('len(stream) < 3')
            source_stream = None

        # No data found
        if source_stream is None:
            return None

        # Generate streams
        sample_list = cls.augmented_set(event, network_code, station, source_stream, inventory,
                                        num_samples)
        return sample_list

    def estimate_arrival_time(self):
        """Estimates the time of P-wave arrival from the event origin to the station.

        Returns
        -------
        arrival : `obspy.UTCDateTime`
            The estimated time of arrival.
        """

        # Estimate arrivals
        arrivals = _VELOCITY_MODEL.get_travel_times_geo(
            self.depth, self.event.latitude, self.event.longitude,
            self.station.latitude, self.station.longitude,
            phase_list=['p', 'P'])

        logger.debug(arrivals)

        # Get travel time from arrivals
        travel_time = arrivals[0].time

        # Compute arrival time
        arrival_time = self.event.time + travel_time
        return arrival_time

    def trace_signal_to_noise(self, trace):
        """Computes the signal-to-noise ratio of the trace.

        Parameters
        ----------
        trace : `obspy.Trace`
            The queried trace.

        Returns
        -------
        snr : `float`
            Signal-to-noise ratio.
        """

        # Determine the time to split the trace
        start_snr_window = self.eta - 0.5
        snr_window_length = 7

        # Compute the SNR
        signal_slice = trace.slice(starttime=start_snr_window,
                                   endtime=start_snr_window + snr_window_length)
        noise_slice = trace.slice(endtime=start_snr_window)

        signal_std = np.absolute(signal_slice.max())
        noise_std = np.absolute(noise_slice.max())
        return signal_std / noise_std

    def signal_to_noise_ratio(self):
        """Computes the signal-to-noise ratio of the filtered stream.

        Returns the maximum signal-to-noise ratio among the stream's traces.

        Returns
        -------
        snr : `float`
            Signal-to-noise ratio.
        """

        return max([self.trace_signal_to_noise(trace) for trace in self.filtered_stream])

    def add_noise(self):
        """Adds random normal noise to the stream.

        Scales the noise relatively to `self.stream_max`, so should be used before normalization."""

        for trace in self.filtered_stream:
            trace.data = trace.data + np.random.normal(0.0,
                                                       0.1 * self.stream_max * random(),
                                                       len(trace))

    def save_tfrecords(self, base_dir, relative_path, save_raw=False):
        """Save .tfrecords files relative to the `filtered_stream` (and
        optionally also the `raw_stream`) to a path given by
        `base_dir` + `'filter/'` or `'raw/'` + `relative_path`.
        """

        filtered_path = os.path.join(base_dir, 'filter', relative_path)
        self._save_tfrecords_file(self.filtered_stream, self.stream_max, filtered_path,
                                  event_type=self.event.type_code(), distance=self.distance,
                                  magnitude=self.magnitude, depth=self.depth, azimuth=self.azimuth)
        if save_raw:
            raw_path = os.path.join(base_dir, 'raw', relative_path)
            self._save_tfrecords_file(self.raw_stream, self.stream_max, raw_path,
                                      event_type=self.event.type_code(), distance=self.distance,
                                      magnitude=self.magnitude, depth=self.depth,
                                      azimuth=self.azimuth)
