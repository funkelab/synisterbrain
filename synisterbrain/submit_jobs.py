import argparse
import numpy as np
import os
from tqdm import tqdm

from synistereq.models import KNOWN_MODELS
from synistereq.repositories import KNOWN_REPOSITORIES
from synistereq.utils import log_config
from synisterbrain.loader import get_data_loader
from synisterbrain.brain_db import BrainDb
from funlib.run import run
from subprocess import Popen
import threading

import time
import logging
import torch

def submit_jobs(db_credentials,
                db_name,
                collection_name,
                predict_id,
                dataset,
                model,
                n_gpus,
                n_cpus,
                batch_size,
                prefetch_factor,
                queue,
                singularity_container=None,
                mount_dirs=["/nrs", "/scratch", "/groups", "/misc"]):

    brain_db = BrainDb(db_credentials,
                       db_name,
                       collection_name,
                       predict_id)
    #brain_db.initialize()

    dx = model.input_shape[2] * dataset.voxel_size[2]
    dy = model.input_shape[1] * dataset.voxel_size[1]
    dz = model.input_shape[0] * dataset.voxel_size[0]
    
    job_script = os.path.join(os.path.dirname(__file__), "job.py")
    dataset_name = dataset.name.upper()

    for gpu_id in range(n_gpus):
        base_cmd = f"python -u {job_script} --creds {db_credentials} --db {db_name} "+\
                   f"--coll {collection_name} --id {predict_id} "+\
                   f"--dat {dataset_name} --gpus {n_gpus} " +\
                   f"--cpus {n_cpus} --bsize {batch_size} --prefetch {prefetch_factor} --gpuid {gpu_id}"

        if queue is not None:
            cmd = run(command=base_cmd,
                      queue=queue,
                      num_gpus=1,
                      num_cpus=n_cpus,
                      singularity_image=singularity_container,
                      mount_dirs=mount_dirs,
                      execute=False,
                      expand=True)
        else:
            cmd = base_cmd
        
        cmd = [c for c in cmd.split(" ") if c != '']
        cmd_string = ""
        for c in cmd:
            cmd_string += str(c) + " "
        print(cmd_string)
        Popen(cmd_string, shell=True)


parser = argparse.ArgumentParser()
parser.add_argument("--log-file", default="whole_volume.log")
parser.add_argument("--creds", help="Db credentials file path")
parser.add_argument("--db", help="DB name")
parser.add_argument("--coll", help="DB collection")
parser.add_argument("--predict-id", help="ID of prediction", type=int)
parser.add_argument(
    "--repository", help=", ".join(KNOWN_REPOSITORIES.keys()), type=str, default=None,
)
parser.add_argument("--queue", help="LSF queue name", default="gpu_rtx")
parser.add_argument("--n-gpus", help="Total number of GPUs/jobs to submit (1 GPU per job)", type=int, default=2)
parser.add_argument("--n-cpus", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=16)


if __name__ == "__main__":
    args = parser.parse_args()
    repo = KNOWN_REPOSITORIES[args.repository]()
    model = KNOWN_MODELS[repo.dataset.name.upper()]()

    log_config(args.log_file)
    submit_jobs(
            db_credentials=args.creds,
            db_name=args.db,
            collection_name=args.coll,
            predict_id=args.predict_id,
            dataset=repo.dataset,
            model=model,
            n_gpus=args.n_gpus,
            n_cpus=args.n_cpus,
            batch_size=args.batch_size,
            prefetch_factor=20,
            queue=args.queue,
            singularity_container=None,
            mount_dirs=["/nrs", "/scratch", "/groups", "/misc"])

