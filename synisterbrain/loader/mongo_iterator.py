from configparser import ConfigParser
import numpy as np
import zarr
import daisy
import math
from pymongo import MongoClient
import logging
import time
from synistereq.datasets import Fafb

from synisterbrain.brain_db import BrainDb

log = logging.getLogger(__name__)

class MongoIterator(object):
    def __init__(self, 
                 credentials, 
                 db_name, 
                 collection_name,
                 predict_id,
                 dataset,
                 dx,dy,dz,
                 n_gpus,
                 gpu_id,
                 n_cpus,
                 cpu_id,
                 transform=None):

        self.db = BrainDb(credentials, db_name, collection_name, predict_id)

        start = time.time()
        log.info("Partition DB to workers...")
        self.cursor = self.get_cursor(n_gpus, n_cpus, gpu_id, cpu_id) 
        log.info(f"...took {time.time() - start} seconds")

        self.dataset = dataset
        self.data = dataset.open_daisy()

        self.transform = transform

        if dx % 8 != 0 or dy % 8 != 0 or dz % 80 != 0:
            raise ValueError("Roi size must be divisible by two")

        self.dx = dx
        self.dy = dy
        self.dz = dz

    def get_chunks(self, n_elements, k_chunks):
        ch = [(n_elements // k_chunks) + (1 if i < (n_elements % k_chunks) else 0) for i in range(k_chunks)]
        return ch

    def get_cursor(self, n_gpus, n_cpus, gpu_id, cpu_id):
        n_open_documents = self.db.len_not_predicted()
        gpu_chunks = self.get_chunks(n_open_documents, n_gpus)
        print(len(gpu_chunks))
        print(gpu_chunks)
        print(gpu_id)
        gpu_offset = int(np.sum(gpu_chunks[:gpu_id]))
        gpu_len = gpu_chunks[gpu_id]
        
        cpu_chunks = self.get_chunks(gpu_len, n_cpus)
        cpu_offset = int(np.sum(cpu_chunks[:cpu_id]))
        cpu_len = cpu_chunks[cpu_id]

        doc_offset = gpu_offset + cpu_offset
        doc_len = cpu_len

        log.info(f"Partition ({gpu_id}, {cpu_id}): Start {doc_offset}, Len {doc_len}")
        print("GPU", gpu_offset, gpu_len)
        print("CPU", cpu_offset, cpu_len)
        print("DOC", doc_offset, doc_len)
        cursor = self.db.get_not_predicted_cursor(doc_offset, doc_len)
        return cursor
 
    def __iter__(self):
        return self

    def __next__(self):
        doc = next(self.cursor, None)
        if doc == None:
            return None
        pre_x = int(doc["pre_x"])
        pre_y = int(doc["pre_y"])
        pre_z = int(doc["pre_z"])
        synapse_id = int(doc["id"])
        center = np.array([pre_z, pre_y, pre_x])
        offset = center - np.array([self.dz/2, self.dy/2, self.dx/2])
        roi = daisy.Roi(offset, (self.dz,self.dy,self.dx))
        array_data = None
        if self.data.roi.contains(roi):
            array = self.data[roi]
            array.materialize()
            array_data = array.data.astype(np.float32)
        elif self.data.roi.intersects(roi):
            array_data = self.data.to_ndarray(roi=roi, fill_value=0).astype(np.float32)

        if array_data is not None:
            array_data = self.dataset.normalize(array_data)

        if self.transform is not None and array_data is not None:
            array_data = self.transform(array_data)

        return {"id": synapse_id, "data": array_data} 

if __name__ == "__main__":
    mongo_em = MongoIterator("/groups/funke/home/ecksteinn/Projects/synex/synister/db_credentials.ini",
                             "synful_synapses",
                             "partners",
                             3,
                             Fafb(),
                             400,
                             400,
                             80,
                             1,0,1,0)

    print(Fafb().voxel_size)

    i = 0
    for doc in mongo_em:
        print(doc)
        i += 1
        if i > 2: break
