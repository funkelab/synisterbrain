import argparse
from tqdm import tqdm

from synistereq.repositories import KNOWN_REPOSITORIES, Repository
from synisterbrain.brain_db import BrainDb


def ingest(db: BrainDb, repo: Repository, resume_offset=None):
    db.write_state_metadata(repo.service.get_state_metadata())

    for batch in tqdm(
        repo.service.pre_synapse_batches(resume_offset=resume_offset),
        desc="Importing synapses",
    ):
        positions = batch[[*"xyz"]].values
        positions = repo.transform_positions(positions)
        batch[[*"xyz"]] = positions

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


if __name__ == "__main__":
    args = parser.parse_args()

    db = BrainDb(args.creds, args.db, args.coll, None)
    repo = KNOWN_REPOSITORIES[args.repository]()

    if args.overwrite:
        db.drop_collection()
        db.initialize()

    ingest(db, repo, args.resume)
