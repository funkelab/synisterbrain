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

    def create(self, n_gpus, n_cpus, overwrite=False):
        db = self.__get_db()

        if overwrite:
            db.drop_collection(self.collection)

        # Synapses
        coll = db[self.collection]
        coll.create_index([("id", ASCENDING)],
                            name="id",
                            unique=True)

        
        coll_meta = db[self.meta_collection]
        worker_docs = []
        for i in range(n_gpus):
            for k in range(n_cpus):
                worker_doc = {"gpu_id": i, "cpu_id": k, "max_cursor_id": 0}
                coll_meta.update_one({"$and":[{"gpu_id": worker_doc["gpu_id"]}, {"cpu_id": worker_doc["cpu_id"]}]},
                                     {"$setOnInsert": worker_doc},
                                      upsert=True)
                #coll_meta.insert_many(worker_docs)

    def update_meta(self, gpu_id, cpu_id, max_cursor_id):
        db = self.__get_db()
        coll_meta = db[self.meta_collection]
        coll_meta.update_one({"$and":[{"gpu_id": gpu_id}, {"cpu_id": cpu_id}]}, 
                             {"$set": {"max_cursor_id": max_cursor_id}}, 
                             upsert=False)

    def get_max_cursor_ids(self):
        db = self.__get_db()
        coll_meta = db[self.meta_collection]
        max_cursor_ids = {}
        cursor = coll_meta.find({})
        for doc in cursor:
            max_cursor_ids[(doc["gpu_id"], doc["cpu_id"])] = doc["max_cursor_id"]
        return max_cursor_ids

    def write_predictions(self, predictions):
        coll = self.__get_collection()
        coll.insert_many(predictions)

    def validate_ids(self, delete_duplicates=False):
        coll = self.__get_collection()
        q = coll.aggregate([
          { "$group": {
            "_id": { "id": "$id" }, 
            "uniqueIds": { "$addToSet": "$_id" },
            "count": { "$sum": 1 } 
          } }, 
          { "$match": { 
            "count": { "$gte": 2 } 
          } },
          { "$sort" : { "count" : -1} },
        ], allowDiskUse=True)
        non_unique_docs = [v for v in q]
        for v in non_unique_docs:
            print(v["_id"]["id"])
        print("N not unique:", len(non_unique_docs))

        print("Delete...")
        if delete_duplicates:
            for v in non_unique_docs:
                coll.delete_one({"_id": v["uniqueIds"][0]})

