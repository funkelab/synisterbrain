from torch.utils.data import IterableDataset, DataLoader
from configparser import ConfigParser
import zarr
import daisy
import pymongo
from pymongo import MongoClient
import torch
from synisterbrain.loader import MongoIterator
from synistereq.datasets import Fafb

import logging
import time

log = logging.getLogger(__name__)

def get_data_loader(db_credentials,
                    db_name,
                    collection_name,
                    dataset,
                    dx, dy, dz,
                    n_gpus,
                    gpu_id,
                    n_cpus,
                    batch_size,
                    prefetch_factor, 
                    max_cursor_ids):

    log.info(f"Initialize data loader {gpu_id+1}/{n_gpus}...")
    log.info(f"with {n_cpus} cpus, {batch_size} batch size, prefetch {prefetch_factor}")

    mongo_em_dataset = MongoEM(db_credentials,
                               db_name,
                               collection_name,
                               dataset,
                               dx,
                               dy,
                               dz,
                               n_gpus=n_gpus,
                               gpu_id=gpu_id,
                               max_cursor_ids=max_cursor_ids)

    def collate_fn(batch):
        batch_data = {"id": [b["id"] for b in batch], 
                      "data": torch.cat([b["data"] for b in batch], dim=0), 
                      "cursor_id": [b["cursor_id"] for b in batch],
                      "gpu_id": [b["gpu_id"] for b in batch],
                      "cpu_id": [b["cpu_id"] for b in batch]}
        return batch_data
    
    mongo_em_data_loader = DataLoader(mongo_em_dataset, 
                                      batch_size=batch_size,
                                      pin_memory=True,
                                      collate_fn=collate_fn,
                                      num_workers=n_cpus,
                                      prefetch_factor=prefetch_factor)

    return mongo_em_data_loader


class MongoEM(IterableDataset):
    def __init__(self, 
                 db_credentials,
                 db_name,
                 collection_name,
                 dataset,
                 dx,
                 dy,
                 dz,
                 n_gpus=1,
                 gpu_id=0,
                 max_cursor_ids=None):

        log.info(f"Initialize dataset {gpu_id+1}/{n_gpus}...")
        self.db_name = db_name
        self.collection_name = collection_name
        self.db_credentials = db_credentials
        self.dataset = dataset
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.n_gpus = n_gpus
        self.gpu_id = gpu_id
        self.max_cursor_ids = max_cursor_ids

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            log.info("No worker info available, use single process data loading.")
            return MongoIterator(self.db_credentials,
                                 self.db_name,
                                 self.collection_name,
                                 self.dataset,
                                 self.dx,
                                 self.dy,
                                 self.dz,
                                 self.n_gpus,
                                 self.gpu_id,
                                 n_cpus=1,
                                 cpu_id=0,
                                 max_cursor_ids=self.max_cursor_ids,
                                 transform=self.transform_to_tensor)
        else:
            n_cpus = int(worker_info.num_workers)
            cpu_id = int(worker_info.id)
            log.info("Worker info available, use multiprocess data loading.")
            log.info(f"Init cpu {cpu_id+1}/{n_cpus}...")
            return MongoIterator(self.db_credentials,
                                 self.db_name,
                                 self.collection_name,
                                 self.dataset,
                                 self.dx,
                                 self.dy,
                                 self.dz,
                                 self.n_gpus,
                                 self.gpu_id,
                                 n_cpus=n_cpus,
                                 cpu_id=cpu_id,
                                 max_cursor_ids=self.max_cursor_ids,
                                 transform=self.transform_to_tensor)

    def transform_to_tensor(self, data_array):
        tensor_array = torch.tensor(data_array)
        # Add channel and batch dim:
        tensor_array = tensor_array.unsqueeze(0).unsqueeze(0)
        return tensor_array


if __name__ == "__main__":
    mongo_em = MongoEM("/groups/funke/home/ecksteinn/Projects/synex/synister/db_credentials.ini",
                       "synful_synapses",
                       "partners",
                       Fafb(),
                       400,
                       400,
                       80)

    i = 0
    for doc in mongo_em:
        print(doc[0])
        print(doc[1].shape)
        i += 1
        if i > 2: break
