from pymongo import MongoClient, IndexModel, ASCENDING
from configparser import ConfigParser

class BrainDb(object):
    def __init__(self, credentials, db_name, collection_name, predict_id):
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
        self.predict_id = predict_id
        self.predicted_field = f"predicted_{self.predict_id}"
        self.nt_field = f"nts_{self.predict_id}"
        self.meta_collection = collection_name + "_meta"

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

    def initialize(self):
        coll = self.__get_collection()
        self.reset()
        coll.update_many({}, {"$set": {self.predicted_field: False}}, upsert=False)

    def reset(self):
        coll = self.__get_collection()
        coll.update_many({}, {"$unset": {self.predicted_field: 1, self.nt_field: 1}})

    def write_predictions(self, predictions):
        coll = self.__get_collection()
        for prediction in predictions:
            coll.update_one({"id": prediction["id"]}, {"$set": {self.nt_field: prediction["nts"],
                                                                self.predicted_field: True}}, 
                                                                upsert=False)

    def get_not_predicted_cursor(self, doc_offset, doc_len):
        coll = self.__get_collection()
        cursor = coll.find({self.predicted_field: False},
                            no_cursor_timeout=True).skip(doc_offset).limit(doc_len)
        return cursor

    def count_docs(self, query):
        coll = self.__get_collection()
        n_docs = coll.count_documents(query)
        return n_docs

if __name__ == "__main__":
    predict_id = 3
    db_name = "synful_synapses"
    collection_name = "partners"
    credentials = "/groups/funke/home/ecksteinn/Projects/synex/synisterbrain/db_credentials.ini"
    db = BrainDb(credentials, db_name, collection_name, predict_id)
    db.initialize()
    #db.write_prediction({"id": 0, "nts": {"ach": 0.8, "gaba": 0.01, "dop": 0.3}})
