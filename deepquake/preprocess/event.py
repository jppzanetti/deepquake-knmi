import os
import glob
import random
import logging

from obspy.core.event.catalog import read_events
import obspy.geodetics.base as geo


CATALOG_FILENAME = 'inputdata/event_catalog.xml'


logger = logging.getLogger('event.py')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


class Event:
    """Stores the information of an event.

    Parameters
    ----------
    latitude : `float`
        Event latitude.
    longitude : `float`
        Event longitude.
    depth : `float`
        Event depth in km.
    time : `obspy.UTCDateTime`
        Time of event.
    magnitude : `float`
        Event magnitude.
    event_type : `str`
        Type of event.
    event_id : `str`
        Event identifier.

    Attributes
    ----------
    latitude : `float`
        Event latitude.
    longitude : `float`
        Event longitude.
    depth : `float`
        Event depth in km.
    time : `obspy.UTCDateTime`
        Time of event.
    magnitude : `float`
        Event magnitude.
    event_type : `str`
        Type of event.
    event_id : `str`
        Event identifier.
    """

    _EVENT_TYPE_CODES = {
        'induced or triggered event': 1,
        'earthquake': 1,
        'other event': 2,
        'not existing': 0,
        'explosion': 2,
        'mine collapse': 2,
        'sonic boom': 3,
        'quarry blast': 2
    }

    def __init__(self, latitude, longitude, depth, time, magnitude, event_type, event_id):
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.time = time
        self.magnitude = magnitude
        self.event_type = event_type
        self.event_id = event_id

    @classmethod
    def from_obspy_event(cls, event):
        """Creates an Event object from an obspy.Event."""
        origin = event.preferred_origin()

        magnitude = -1.0
        if event.preferred_magnitude() is not None:
            magnitude = event.preferred_magnitude().mag

        return cls(origin.latitude, origin.longitude, origin.depth / 1000.0,
                   origin.time,
                   magnitude, event.event_type, str(event.resource_id))

    def __str__(self):
        return '{} ({:+.3f}, {:+.3f}) Type: {} Depth: {:.2f} Magnitude: {:.2f}'.format(
            self.time.isoformat(), self.latitude, self.longitude,
            self.type_code(), self.depth, self.magnitude)

    def type_code(self):
        """Returns None if the event has no type assigned, 0 if 'not
        existing', 1 if it is a seismic event, or 2 if it's another type of
        event."""

        if self.event_type is None:
            return None
        return self._EVENT_TYPE_CODES[self.event_type]

    def distance_to(self, station):
        """Return the distance (in km) between the event and the given station."""

        dist_meters, _, _ = geo.gps2dist_azimuth(station.latitude,
                                                 station.longitude,
                                                 self.latitude,
                                                 self.longitude,
                                                 a=geo.WGS84_A, f=geo.WGS84_F)
        return dist_meters / 1000.0


def read_catalog_directory(event_dir, event_fraction):
    """Read the catalog from a directory of catalogs per station and save it to a dictionary.

    Parameters
    ----------
    event_dir : `str`
        The directory of events.
    event_fraction : `float`
        Fraction of events to use.

    Returns
    -------
    catalog : `dict` (`str`, `str`) -> `list` of `Event`
        A dictionary where the keys are a pair of strings (network, station),
        and the values are a list of the events associated with that station.
    all_events : `list` of `Event`
        A list of all the events in the catalog.
    """

    catalog = {}
    all_events = []
    for catalog_filename in glob.glob(os.path.join(event_dir, '*.xml')):
        # Get network and station codes from filename
        basename = os.path.basename(catalog_filename)
        tokens = basename.split('_')
        network_code = tokens[0]
        station_code = tokens[1]

        if (network_code, station_code) not in catalog:
            catalog[(network_code, station_code)] = []

        event_list = [Event.from_obspy_event(event)
                      for event in read_events(catalog_filename).events]
        num_events_catalog = len(event_list)
        all_events.extend(event_list)
        if event_fraction < 1.0:
            num_events = int(num_events_catalog * event_fraction)
            event_list = random.sample(event_list, num_events)
        logger.info('%s Using %d / %d events', basename, len(event_list), num_events_catalog)
        catalog[(network_code, station_code)].extend(event_list)

    return catalog, all_events


def read_catalog_file(event_fraction, station_dict):
    """Read a catalog file containing all events, associate them to stations,
    and save the results to a dictionary.

    Parameters
    ----------
    event_fraction : `float`
        Fraction of events to use.
    station_dict : `dict` (`str`, `str`) -> `obspy.Station`
        A dictionary pointing a pair (`network_code`, `station_code`)
        to the respective `Station` object.

    Returns
    -------
    catalog : `dict` (`str`, `str`) -> `list` of `Event`
        A dictionary where the keys are a pair of strings (network, station),
        and the values are a list of the events associated with that station.
    all_events : `list` of `Event`
        A list of all the events in the catalog.
    """

    # Initialization
    all_events = [Event.from_obspy_event(event)
                  for event in read_events(CATALOG_FILENAME).events]
    catalog = {(network, station): [] for network, station in station_dict}

    # Filter non-events out
    all_events = [event for event in all_events
                  if event.type_code is not None and event.type_code() > 0]

    # Associate events with stations using magnitude and distance
    max_distance = 50
    for event in all_events:
        for network, station in catalog:
            # Check if station is active
            if not station_dict[(network, station)].is_active(time=event.time):
                continue

            # Check distance to event
            dist = event.distance_to(station_dict[(network, station)])
            #logger.debug('%s %s %.2f', station, event.time.isoformat(), dist)
            if dist > max_distance:
                continue
            if (event.type_code() == 1
                    and event.magnitude is not None and event.magnitude <= 2.5
                    and dist > 0.5 * max_distance):
                continue
            if (event.type_code() == 1
                    and event.magnitude is not None and event.magnitude <= 1.5
                    and dist > 0.2 * max_distance):
                continue

            catalog[(network, station)].append(event)

    # Select event_fraction
    if event_fraction < 1.0:
        for network, station in catalog:
            event_list = []

            type1_events = [ev for ev in catalog[(network, station)] if ev.type_code() == 1]
            num_type1 = int(len(type1_events) * event_fraction)
            event_list = random.sample(type1_events, num_type1)

            # Try to balance type 2 events with the more numerous type 1
            type2_events = [ev for ev in catalog[(network, station)] if ev.type_code() == 2]
            num_type2 = int(len(type2_events) * event_fraction)
            if num_type2 < num_type1:
                num_type2 = min(len(type2_events), num_type1)
            event_list.extend(random.sample(type2_events, num_type2))

            logger.info('%s %s Using %d / %d events',
                        network, station,
                        len(event_list), len(catalog[(network, station)]))
            catalog[(network, station)] = event_list

    return catalog, all_events
