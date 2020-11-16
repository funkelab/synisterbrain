from pymongo import MongoClient, IndexModel, ASCENDING
from configparser import ConfigParser

class BrainDb(object):
    def __init__(self, credentials, db_name, collection_name):
        with open(credentials) as fp:
            config = ConfigParser()
            config.read_file(fp)
            self.credentials = {}
            self.credentials["user"] = config.get("Credentials", "user")
            self.credentials["password"] = config.get("Credentials", "password")
            self.credentials["host"] = config.get("Credentials", "host")
            self.credentials["port"] = config.get("Credentials", "port")

        self.auth_string = 'mongodb://{}:{}@{}:{}'.format(self.credentials["user"],
                                                          self.credentials["password"],
                                                          self.credentials["host"],
                                                          self.credentials["port"])

        self.db_name = db_name
        self.collection = collection_name

    def __get_client(self):
        client = MongoClient(self.auth_string, connect=False)
        return client

    def __get_db(self, db_name=None):
        if db_name is None:
            db_name = self.db_name
        client = self.__get_client()
        db = client[db_name]
        return db

    def __get_collection(self):
        db = self.__get_db()
        return db[self.collection]

    def create(self, overwrite=False):
        db = self.__get_db()

        if overwrite:
            db.drop_collection(self.collection)

        # Synapses
        coll = db[self.collection]
        coll.create_index([("id", ASCENDING)],
                            name="id",
                            sparse=True)

    def write_predictions(self, predictions):
        coll = self.__get_collection()
        coll.insert_many(predictions)
