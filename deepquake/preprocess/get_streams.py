"""Gets event streams from an FDSN web service."""

import argparse

import obspy
import obspy.clients.fdsn as fdsn
import obspy.geodetics.base as geo

from obspy import UTCDateTime, read_inventory
from obspy.taup import TauPyModel
from obspy.clients.fdsn.header import FDSNNoDataException

import numpy as np
import tensorflow as tf
import os
import glob
import logging
import random

from stream import Stream, EventStream, WINDOW_LENGTH
from event import read_catalog_file

# Workaround to fix logging, might break tf logging
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False


SNR_THRESHOLD = 4.0
MIN_NOISE_INTERVAL = 60.0 * 120.0
NOISE_PADDING = 60.0 * 15.0


# Set up logging
logger = logging.getLogger('get_streams.py')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


def get_event_samples(client, catalog, inventory, station_dict, output_path,
                      augment_copies=0, save_mseed=False, save_raw=False, try_fdsn=False):
    """Gets event samples and saves them to .tfrecords files.

    Parameters
    ----------
    client : `obspy.clients.fdsn.Client`
        A client pointing to the appropriate FDSN web service.
    catalog : `dict` (`str`, `str`) -> `list` of `Event`
        A dictionary of events to lookup for each station.
    inventory : `obspy.Inventory`
        The station inventory.
    station_dict : `dict` (`str`, `str`) -> `obspy.Station`
        A dictionary pointing a pair (`network_code`, `station_code`)
        to the respective `Station` object.
    output_path : `str`
        Path to the output directory.
    augment_copies : `int`
        Number of extra copies for each valid sample, generated by shifting
        the window and adding random normal noise.
    save_mseed : `bool`
        Whether to save a miniSEED backup of the streams. (default `False`)
    save_raw : `bool`
        Whether to save an unfiltered version of the streams. (default `False`)
    try_fdsn : `bool`
        Whether to fetch streams from FDSN if they're not available locally. (default `False`)

    Returns
    -------
    num_samples : `int`
        Number of samples saved.
    """

    num_event_samples = 0
    num_queries = 0

    total_events = sum([len(events) for events in catalog.values()])

    logger.info('# network station time azimuth depth distance magnitude')
    for network_code, station_code in catalog:
        station = station_dict[(network_code, station_code)]
        for event in catalog[(network_code, station_code)]:
            num_queries += 1
            logger.debug('%d/%d %s %s Event: %s',
                         num_queries, total_events, network_code, station_code, event)

            # Check if station is active
            if not station.is_active(time=event.time):
                logger.info('%d/%d Station %s not active at %s',
                            num_queries, total_events, station_code, event.time.isoformat())
                logger.debug('\n\n')
                continue

            # Name of the file without extension
            base_filename = (network_code + '_' + station.code + '_'
                             + event.time.strftime('%Y-%m-%dT%H%M%S'))

            # Fetch stream
            try:
                mseed_path = os.path.join(output_path, 'mseed/', str(event.type_code()),
                                          base_filename + '.mseed')
                sample_list = EventStream.from_local_sds(event, network_code, station, inventory,
                                                         num_samples=(1 + augment_copies))

                if sample_list is None and try_fdsn:
                    sample_list = EventStream.from_fdsn(
                        client, event, network_code, station, inventory,
                        num_samples=(1 + augment_copies),
                        mseed_path=mseed_path, save_mseed=save_mseed
                    )
                if sample_list is None:
                    continue

                stream = sample_list[0]
                logger.debug(stream.raw_stream)
                logger.debug('ETA: %s', stream.eta.isoformat())
            except ValueError:
                logger.info('%d/%d Invalid stream', num_queries, total_events)
                logger.debug('\n\n')
                continue
            except FDSNNoDataException:
                logger.error('%d/%d No data available', num_queries, total_events)
                logger.debug('\n\n')
                continue
            except AttributeError as e:
                logger.exception(str(e))
                continue
            except NotImplementedError:
                logger.debug('%d/%d NotImplementedError probably rotation in preprocessing',
                             num_queries, total_events)
                continue

            # Check signal-to-noise ratio
            snr = stream.signal_to_noise_ratio()
            logger.debug('SNR: %.2f', snr)
            if snr < SNR_THRESHOLD:
                logger.info('%d/%d Stream with low signal-to-noise ratio: %.2f',
                            num_queries, total_events, snr)
                logger.debug('\n\n')
                continue

            # Save .tfrecords files for both streams
            try:
                stream.save_tfrecords(output_path,
                                      os.path.join(str(event.type_code()),
                                                   base_filename + '.tfrecords'),
                                      save_raw)
            except ValueError as e:
                logger.exception(e)
                continue

            num_event_samples += 1
            logger.info('%d/%d %d %s %s %s %.2f %.2f %.2f %.2f %.2f',
                        num_queries, total_events,
                        num_event_samples, network_code, station_code, event.time.isoformat(),
                        stream.azimuth, stream.depth, stream.distance, stream.magnitude,
                        snr)
            logger.debug('\n\n')

            # Save extra samples
            if augment_copies > 0:
                for copy_num, stream in enumerate(sample_list[1:]):
                    base_filename = (network_code + '_' + station.code + '_'
                                     + event.time.strftime('%Y-%m-%dT%H%M%S') + '_'
                                     + str(copy_num) + '.tfrecords')

                    try:
                        stream.save_tfrecords(output_path,
                                              os.path.join(str(event.type_code()), base_filename),
                                              save_raw)
                    except ValueError:
                        logger.error('Error saving %s', base_filename)
                        continue

    return num_event_samples


def get_noise_samples_single(client, all_events, inventory, station_dict, num_samples, output_path,
                             save_raw=False):
    """Gets `num_samples` noise samples and saves them to .tfrecords files.

    Parameters
    ----------
    client : `obspy.clients.fdsn.Client`
        A client pointing to the appropriate FDSN web service.
    all_events : `list` of `Event`
        A list containing all cataloged events.
    inventory : `obspy.Inventory`
        The station inventory.
    station_dict : `dict` (`str`, `str`) -> `obspy.Station`
        A dictionary pointing a pair (`network_code`, `station_code`)
        to the respective `Station` object.
    num_samples : `int`
        Number of desired samples.
    output_path : `str`
        Path to the output directory.
    save_raw : `bool`
        Whether to save an unfiltered version of the streams.
    """
    # Sort events by time
    all_events.sort(key=lambda e: e.time)

    # Get all intervals between events that are long enough
    intervals = []
    for i in range(len(all_events) - 1):
        t0 = all_events[i].time
        t1 = all_events[i + 1].time

        if t1 - t0 > MIN_NOISE_INTERVAL:
            intervals.append((t0 + NOISE_PADDING, t1 - NOISE_PADDING))
    logger.debug('Number of noise intervals: %d', len(intervals))

    nodata_stations_per_interval = [[] for _ in range(len(intervals))]
    network_station_list = list(station_dict.keys())
    num_noise = 0
    while num_noise < num_samples:
        # Take random interval and station
        rand_interval_idx = random.randrange(len(intervals))
        t0, t1 = intervals[rand_interval_idx]
        network_code, station_code = random.choice(network_station_list)

        # Check whether it has been tried
        if (network_code, station_code) in nodata_stations_per_interval[rand_interval_idx]:
            logger.error('%s %s %d Stored no data', network_code, station_code, rand_interval_idx)
            continue

        # Determine random moment inside interval
        time = t0 + random.random() * (t1 - t0)

        # Fetch stream
        try:
            logger.debug('%s %s %s %s %s', network_code, station_code, t0, t1, time)
            stream = Stream.from_fdsn(client, network_code, station_code, time, inventory)

            # Local option still does not work
            # stream = Stream.from_local_sds(network_code, station_code, time, inventory)
        except FDSNNoDataException:
            nodata_stations_per_interval[rand_interval_idx].append((network_code, station_code))
            logger.error('%s %s %d No data available', network_code, station_code, rand_interval_idx)
            logger.debug('\n\n')
            continue
        except ValueError:
            logger.info('Invalid stream')
            logger.debug('\n\n')
            continue
        except NotImplementedError:
            logger.exception('%s %s %d Blame obspy for this one',
                             network_code, station_code, rand_interval_idx)
            continue

        # Validate stream
        if stream is None:
            logger.info('Null stream')
            logger.debug('\n\n')
            continue

        logger.debug(stream.raw_stream)

        # Check signal-to-noise ratio (?)

        # Save .tfrecords for raw and filtered samples
        base_filename = (network_code + '_' + station_code + '_'
                         + time.strftime('%Y-%m-%dT%H%M%S') + '.tfrecords')

        try:
            stream.save_tfrecords(output_path,
                                  os.path.join('0/', base_filename),
                                  save_raw)
        except ValueError:
            logger.error('Error saving %s', base_filename)
            continue

        num_noise += 1
        logger.info('%d/%d %s %s %s',
                    num_noise, num_samples, network_code, station_code, time.isoformat())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=str,
                        help='File containing a FDSNStationXML list of stations to retrieve')
    parser.add_argument('--events', type=str,
                        help='Path to the directory that contains the events')
    parser.add_argument('--output', type=str,
                        help='Path to the output directory')
    parser.add_argument('--event_fraction', type=float,
                        help='Fraction of events to use')
    parser.add_argument('--extra_samples', type=int, default=0,
                        help='Number of extra samples to generate for each event')
    parser.add_argument('--save_mseed', action='store_true',
                        help='Whether to save a miniSEED backup of the streams')
    parser.add_argument('--use-fdsn', action='store_true',
                        help='Try to fetch streams from FDSN if they are not available locally')
    args = parser.parse_args()

    client = fdsn.Client('KNMI')

    # Store station metadata
    inventory = read_inventory(args.inventory)
    station_dict = {}
    for network in inventory.networks:
        for station in network.stations:
            station_dict[(network.code, station.code)] = station

    # Read event catalog
    catalog, all_events = read_catalog_file(args.event_fraction, station_dict)

    # Count events
    sum_events = 0
    num_stations = 0
    for network_code, station_code in catalog:
        num_events = len(catalog[(network_code, station_code)])
        logger.debug('%s %s --- %d events', network_code, station_code, num_events)
        sum_events += num_events
        num_stations += 1
    logger.info('Events: %d / %d Stations: %d', sum_events, len(all_events), num_stations)

    # Create directories if needed
    for version in ['mseed', 'raw', 'filter']:
        for sample_type in [str(event_type) for event_type in range(4)]:
            dirname = os.path.join(args.output, version, sample_type, '')
            if not os.path.exists(dirname):
                os.makedirs(dirname)

    # Get events
    num_event_samples = get_event_samples(client, catalog, inventory, station_dict,
                                          args.output, augment_copies=args.extra_samples,
                                          save_mseed=args.save_mseed, try_fdsn=args.use_fdsn)
    logger.info('Saved %d event samples', num_event_samples)

    # Get noise
    get_noise_samples_single(client, all_events, inventory, station_dict,
                             num_event_samples * (1 + args.extra_samples),
                             args.output)


if __name__ == '__main__':
    main()