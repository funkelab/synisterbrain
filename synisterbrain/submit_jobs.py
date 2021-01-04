import numpy as np
import os
from tqdm import tqdm

from synistereq.datasets import Fafb, Hemi
from synistereq.models import FafbModel, HemiModel
from synistereq.interfaces import Catmaid, Neuprint, Flywire
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
                db_name_read,
                collection_name_read,
                db_name_write,
                collection_name_write,
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
                       db_name_write,
                       collection_name_write)
    brain_db.create(n_gpus, n_cpus)

    dx = model.input_shape[2] * dataset.voxel_size[2]
    dy = model.input_shape[1] * dataset.voxel_size[1]
    dz = model.input_shape[0] * dataset.voxel_size[0]
    
    job_script = os.path.join(os.path.dirname(__file__), "job.py")
    dataset_name = dataset.name

    for gpu_id in range(n_gpus):
        base_cmd = f"python -u {job_script} --creds {db_credentials} --dbr {db_name_read} "+\
                   f"--collr {collection_name_read} --dbw {db_name_write} --collw {collection_name_write} "+\
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
            #cmd = base_cmd.split(" ")
            cmd = base_cmd
        
        #cmd = [c.replace('"',"") for c in cmd.split(" ")]

        cmd = [c for c in cmd.split(" ") if c != '']
        #print(*cmd)
        cmd_string = ""
        for c in cmd:
            cmd_string += str(c) + " "
        print(cmd_string)
        Popen(cmd_string, shell=True)

if __name__ == "__main__":
    log_config("brain.log")
    submit_jobs(db_credentials="/groups/funke/home/ecksteinn/Projects/synex/synisterbrain/db_credentials.ini",
                db_name_read="synful_synapses",
                collection_name_read="partners",
                db_name_write="synful_predictions",
                collection_name_write="predictions_v2",
                dataset=Fafb(),
                model=FafbModel(),
                n_gpus=20,
                n_cpus=5,
                batch_size=8,
                prefetch_factor=10,
                queue="gpu_any",
                singularity_container=None,
                mount_dirs=["/nrs", "/scratch", "/groups", "/misc"])

