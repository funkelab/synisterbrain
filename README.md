# Synisterbrain
A package to facilitate the prediction of neurotransmitters in whole brain scale synapse datasets, e.g. all automatically predicted Buhmann Synapses in FAFB (~240,000,000). This is donee by iterating in parallel over a large mongoDB collection and distributing this to multiple GPU workers.

## Installation
```
conda create -n synisterbrain python=3.8 numpy scipy cython
conda activate synisterbrain
pip install -r requirements.txt
pip install .
```

## Usage
Requires a database collection holding all synaptic locations that are to be predicted, e.g. a mongo import of Buhmann synapses. For an example call to start predictions see `synisterbrain/submit_jobs.py`

Start via:
```
python synisterbrain/submit_jobs.py
```
