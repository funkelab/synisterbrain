from configparser import ConfigParser
import numpy as np
import zarr
import daisy
import math
from pymongo import MongoClient

import logging
import time
log = logging.getLogger(__name__)

class MongoIterator(object):
    def __init__(self, 
                 credentials, 
                 db_name, 
                 collection_name,
                 dataset,
                 dx,dy,dz,
                 n_gpus,
                 gpu_id,
                 n_cpus,
                 cpu_id,
                 transform=None):

        log.info(f"Connect to {db_name}/{collection_name}...")
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
        self.collection_name = collection_name
        self.collection = self.__get_collection()

        start = time.time()
        log.info("Partition DB to workers...")
        self.n_gpus = n_gpus
        self.gpu_id = gpu_id
        self.n_documents = self.collection.count_documents({})
        self.gpu_offset = int(math.ceil(float(self.gpu_id)/self.n_gpus * self.n_documents))
        self.gpu_len = int(math.ceil(1./self.n_gpus * self.n_documents))

        self.cpu_id = cpu_id
        self.n_cpus = n_cpus
        self.cpu_offset = int(math.ceil(float(self.cpu_id)/self.n_cpus * self.gpu_len))
        self.cpu_len = int(math.ceil(1./self.n_cpus * self.gpu_len))
        self.doc_offset = self.gpu_offset + self.cpu_offset
        self.doc_len = self.cpu_len
        log.info(f"Partition ({self.gpu_id}, {self.cpu_id}): Start {self.doc_offset}, Len {self.doc_len}")
        self.cursor = self.collection.find({}).skip(self.doc_offset).limit(self.doc_len)
        log.info(f"...took {time.time() - start} seconds")

        self.dataset = dataset
        self.container = dataset.container
        self.dset = dataset.dataset
        self.voxel_size = dataset.voxel_size
        self.data = daisy.open_ds(self.container,
                                  self.dset)
        self.transform = transform

        if dx % 80 != 0 or dy % 8 != 0 or dz % 8 != 0:
            raise ValueError("Roi size must be divisible by two")

        self.dx = dx
        self.dy = dy
        self.dz = dz

    def __get_client(self):
        client = MongoClient(self.auth_string, connect=False)
        return client

    def __get_db(self):
        client = self.__get_client()
        db = client[self.db_name]
        return db

    def __get_collection(self):
        db = self.__get_db()
        return db[self.collection_name]

    def __iter__(self):
         return self

    def __next__(self):
        doc = next(self.cursor, None)
        pre_x = int(doc["pre_x"])
        pre_y = int(doc["pre_y"])
        pre_z = int(doc["pre_z"])
        synapse_id = int(doc["id"])
        center = np.array([pre_z, pre_y, pre_x])
        offset = center - np.array([self.dz/2, self.dy/2, self.dx/2])
        roi = daisy.Roi(offset, (self.dz,self.dy,self.dx))
        array = self.data[roi]
        array.materialize()
        array_data = array.data.astype(np.float32)
        array_data = self.dataset.normalize(array_data)

        if self.transform is not None:
            array_data = self.transform(array_data)
        return {"id": synapse_id,"data": array_data}

if __name__ == "__main__":
    mongo_em = MongoIterator("/groups/funke/home/ecksteinn/Projects/synex/synister/db_credentials.ini",
                             "synful_synapses",
                             "partners",
                             Fafb(),
                             400,
                             400,
                             80)

    i = 0
    for doc in mongo_em:
        print(np.shape(doc))
        i += 1
        if i > 2: break
