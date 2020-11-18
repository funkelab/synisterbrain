import argparse
import time
from tqdm import tqdm
import numpy as np

from synisterbrain.loader import get_data_loader
from synisterbrain.brain_db import BrainDb
from synistereq.models import FafbModel, HemiModel
from synistereq.datasets import Fafb, Hemi
from synistereq.utils import log_config
import torch

import logging
log = logging.getLogger(__name__)

models = {"FAFB": (Fafb, FafbModel), "HEMI": (Hemi, HemiModel)}

parser = argparse.ArgumentParser()
parser.add_argument('--creds', help='Db credentials file path')
parser.add_argument('--dbr', help='DB name to read locs from')
parser.add_argument('--collr', help='DB collection name to read locs from')
parser.add_argument('--dbw', help='DB name to write preds to')
parser.add_argument('--collw', help='DB collection name to wrie preds to')
parser.add_argument('--dat', help='Dataset name corresponding to locations')
parser.add_argument('--cpus', help='Number of cpus each job has access to', type=int)
parser.add_argument('--bsize', help='Data loader batch size', type=int)
parser.add_argument('--prefetch', help='Prefetch factor', type=int)
parser.add_argument('--gpuid', help='Gpu id of the job', type=int)
parser.add_argument('--gpus', help='Total number of GPUs', type=int)

def predict(db_credentials,
            db_name_read,
            collection_name_read,
            db_name_write,
            collection_name_write,
            dataset,
            model,
            gpu_id,
            n_gpus,
            n_cpus,
            batch_size,
            prefetch_factor):

    log.info(f"Prepare GPU {gpu_id}...")
    log.info(f"Connect worker {gpu_id} to db {db_name_write}.{collection_name_write}")
    braindb = BrainDb(db_credentials,
                      db_name_write,
                      collection_name_write)

    max_cursor_ids = braindb.get_max_cursor_ids()

    dx = model.input_shape[2] * dataset.voxel_size[2]
    dy = model.input_shape[1] * dataset.voxel_size[1]
    dz = model.input_shape[0] * dataset.voxel_size[0]

    log.info(f"Get data loader...")
    data_loader = get_data_loader(db_credentials,
                                  db_name_read,
                                  collection_name_read,
                                  dataset,
                                  dx,dy,dz,
                                  n_gpus,
                                  gpu_id,
                                  n_cpus,
                                  batch_size,
                                  prefetch_factor,
                                  max_cursor_ids)

    torch_model = model.init_model()
    torch_model.eval()

    log.info(f"Start prediction...")
    nt_probabilities = []
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, sample in enumerate(tqdm(data_loader)):
        ids = sample['id']
        data = sample['data']
        cursor_ids = sample["cursor_id"]
        gpu_ids = sample["gpu_id"]
        cpu_ids = sample["cpu_id"]
        data = data.to(device)
        prediction = torch_model(data)
        prediction = model.softmax(prediction)

        batch_prediction = []
        meta_updates = []
        worker_id_to_max_cursor_id = {}
        # Iterate over batch and grab predictions
        for k in range(np.shape(prediction)[0]):
            out_k = prediction[k,:].tolist()
            nt_probability = {model.neurotransmitter_list[i]:
                              out_k[i] for i in range(len(model.neurotransmitter_list))}
            nt_probability["id"] = ids[k]
            # cursor ids are monotonic for each worker:
            worker_id_to_max_cursor_id[(gpu_ids[k], cpu_ids[k])] = cursor_ids[k]
            batch_prediction.append(nt_probability)
        braindb.write_predictions(batch_prediction)
        for worker_id, max_id in worker_id_to_max_cursor_id.items():
            braindb.update_meta(worker_id[0], worker_id[1], max_id)

    total_time = time.time() - start
    log.info(f"Total predict time {total_time}")

if __name__ == "__main__":
    args = parser.parse_args()
    dset_model = models[args.dat]
    dset = dset_model[0]()
    model = dset_model[1]()
    log_config(f"worker_{args.gpuid}.log")

    predict(args.creds,
            args.dbr,
            args.collr,
            args.dbw,
            args.collw,
            dset,
            model,
            args.gpuid,
            args.gpus,
            args.cpus,
            args.bsize,
            args.prefetch)
