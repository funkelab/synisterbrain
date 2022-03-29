import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from synistereq.repositories import KNOWN_REPOSITORIES, Repository
from synisterbrain.brain_db import BrainDb


def ingest(db: BrainDb, repo: Repository, resume_offset=None):
    db.write_state_metadata(repo.service.get_state_metadata())

    ingest_batch_sequence(db, repo, repo.service.pre_synapse_batches(resume_offset=resume_offset))


def ingest_dataframe(db: BrainDb, repo: Repository, df, n_batches=1_000):
    batches = np.array_split(df, n_batches)
    ingest_batch_sequence(db, repo, batches)


def ingest_batch_sequence(db: BrainDb, repo: Repository, batch_sequence):
    for batch in tqdm(
        batch_sequence,
        desc="Importing synapses",
    ):
        positions = batch[[*"zyx"]].values
        positions = repo.transform_positions(positions)
        batch[[*"zyx"]] = positions

        batch.reset_index(inplace=True)
        batch.rename(columns={a: f"pre_{a}" for a in "xyz"}, inplace=True)
        batch.rename(columns={"synapse_id": "id"}, inplace=True)
        batch_records = batch.to_dict("records")

        db.write_synapses(batch_records)


parser = argparse.ArgumentParser()
parser.add_argument("--creds", help="Db credentials file path")
parser.add_argument("--db", help="DB name")
parser.add_argument("--coll", help="DB collection")
parser.add_argument(
    "--repository", help=", ".join(KNOWN_REPOSITORIES.keys()), type=str, default=None,
)
parser.add_argument(
    "--overwrite",
    required=False,
    action="store_true",
    help="Overwrite database collection",
)
parser.add_argument(
    "--resume", required=False, type=int, help="Resume from offset index"
)
parser.add_argument(
    "--feather", required=False, help="Feather dataframe", type=str, default=None,
)


if __name__ == "__main__":
    args = parser.parse_args()

    db = BrainDb(args.creds, args.db, args.coll, None)
    repo = KNOWN_REPOSITORIES[args.repository]()

    if args.overwrite:
        db.drop_collection()
        db.initialize()

    if args.feather is not None:
        df = pd.read_feather(args.feather)
        if "synapse_id" not in df.columns:
            df["synapse_id"] = repo.service._synapse_ids(df)
        ingest_dataframe(db, repo, df)
    else:
        ingest(db, repo, args.resume)
