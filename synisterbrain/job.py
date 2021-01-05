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
import pymongo

import logging
log = logging.getLogger(__name__)

models = {"FAFB": (Fafb, FafbModel), "HEMI": (Hemi, HemiModel)}

parser = argparse.ArgumentParser()
parser.add_argument('--creds', help='Db credentials file path')
parser.add_argument('--db', help='DB name')
parser.add_argument('--coll', help='DB collection')
parser.add_argument('--dat', help='Dataset name corresponding to locations')
parser.add_argument('--cpus', help='Number of cpus each job has access to', type=int)
parser.add_argument('--bsize', help='Data loader batch size', type=int)
parser.add_argument('--prefetch', help='Prefetch factor', type=int)
parser.add_argument('--gpuid', help='Gpu id of the job', type=int)
parser.add_argument('--gpus', help='Total number of GPUs', type=int)
parser.add_argument('--id', help='Predict ID', type=int)


def predict(db_credentials,
            db_name,
            collection_name,
            predict_id,
            dataset,
            model,
            gpu_id,
            n_gpus,
            n_cpus,
            batch_size,
            prefetch_factor):

    start = time.time()
    while True:
        try:
            log.info(f"Prepare GPU {gpu_id}...")
            log.info(f"Connect worker {gpu_id} to db {db_name}.{collection_name}")
            braindb = BrainDb(db_credentials,
                              db_name,
                              collection_name,
                              predict_id)

            dx = model.input_shape[2] * dataset.voxel_size[2]
            dy = model.input_shape[1] * dataset.voxel_size[1]
            dz = model.input_shape[0] * dataset.voxel_size[0]

            log.info(f"Get data loader...")
            data_loader = get_data_loader(db_credentials,
                                          db_name,
                                          collection_name,
                                          predict_id,
                                          dataset,
                                          dx,dy,dz,
                                          n_gpus,
                                          gpu_id,
                                          n_cpus,
                                          batch_size,
                                          prefetch_factor)

            torch_model = model.init_model()
            torch_model.eval()

            log.info(f"Start prediction...")
            nt_probabilities = []
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for i, sample in enumerate(tqdm(data_loader, position=gpu_id, desc=f"GPU {gpu_id}")):
                # Cursor done
                if sample is None:
                    break
                ids = sample['id']
                data = sample['data']
                data = data.to(device)
                prediction = torch_model(data)
                prediction = model.softmax(prediction)

                batch_prediction = []
                # Iterate over batch and grab predictions
                for k in range(np.shape(prediction)[0]):
                    out_k = prediction[k,:].tolist()
                    pred = {}
                    nt_probability = {model.neurotransmitter_list[i]:
                                      out_k[i] for i in range(len(model.neurotransmitter_list))}
                    pred["nts"] = nt_probability
                    pred["id"] = ids[k]
                    batch_prediction.append(pred)

                braindb.write_predictions(batch_prediction)

            total_time = time.time() - start
            break
        except pymongo.errors.CursorNotFound:
            log.info(f"Cursor lost... Restart.")
            pass

    log.info(f"Total predict time {total_time}")

if __name__ == "__main__":
    args = parser.parse_args()
    dset_model = models[args.dat]
    dset = dset_model[0]()
    model = dset_model[1]()
    log_config(f"worker_{args.gpuid}.log")

    predict(args.creds,
            args.db,
            args.coll,
            args.id,
            dset,
            model,
            args.gpuid,
            args.gpus,
            args.cpus,
            args.bsize,
            args.prefetch)
