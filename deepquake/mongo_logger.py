"""Very basic module to save records to a Mongo database."""

import urllib
from pymongo import MongoClient
from local_config import MONGODB_HOST, MONGODB_USER, MONGODB_PASS

_MONGODB_HOST = MONGODB_HOST
_MONGODB_PORT = 27017
_MONGODB_DATABASE = 'tf_results'
_MONGODB_DETECTION_COLLECTION = 'predictions'
_MONGODB_LOG_COLLECTION = 'daily_logs'

_MONGODB_USER = urllib.parse.quote_plus(MONGODB_USER)
_MONGODB_PASSWORD = urllib.parse.quote_plus(MONGODB_PASS)


class MongoManager():
    def __init__(self):
        self.client = None
        self.database = None
        self.detection_collection = _MONGODB_DETECTION_COLLECTION

    def connect(self):
        if self.client is not None:
            return

        self.client = MongoClient(_MONGODB_HOST, _MONGODB_PORT,
                                  username=_MONGODB_USER,
                                  password=_MONGODB_PASSWORD)
        self.database = self.client[_MONGODB_DATABASE]

    def save_result(self, document):
        self.database[self.detection_collection].insert_one(document)

    def find_all_detections(self):
        return self.database[self.detection_collection].find()

    def find_detections(self, query):
        return self.database[self.detection_collection].find(query)

    def count_detections(self, query):
        return self.database[self.detection_collection].count_documents(query)

    def register_success(self, day, station):
        document = {'day': day,
                    'station': station,
                    'status': 'Success'}
        self.database[_MONGODB_LOG_COLLECTION].insert_one(document)


mongo_session = MongoManager()
mongo_session.connect()
