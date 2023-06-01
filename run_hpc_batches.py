import subprocess
import os
import sys
from random import randint
import configs.config as config
import itertools
import time
from datetime import datetime

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)

# Create batch number
b = str(randint(1, 1000))

# Determine time of starting
starttime = time.time()
timestamp = datetime.now().strftime("%y%m%d_%H%M")

# Instantiate with default values
databases = [config.db]
models = [config.m]
traits = [config.t]
rfecv_methods = [config.w_rfecv]
noise_methods = [config.leave_out_noisy]
bfi10_methods = [config.leave_out_bfi10]

databases_info = config.label_feature_indexes

# --- HPC GRID---
# # Overwrite default values
databases = ['spc-egemaps.pkl', 'spc-embeddings.pkl', 'noisy-egemaps.pkl', 'reduced-egemaps.pkl']
models = ['svm_rbf', 'rf', 'knn']
# models = ['svm_rbf']
traits = ['e', 'a', 'c', 'n', 'o']
# traits = ['e']
noise_methods = [True, False]
bfi10_methods = [True, False]

# # Define with what HPC config the batches should be run
HPC_CONFIG = FILE_DIR + "/hpc_configs/run_short.sh"

# Create combinations of run parameters
combinations = [[d, m, t, noise, bfi10, rfecv, timestamp, b] for d, m, t, noise, bfi10, rfecv in itertools.product(databases, models, traits, noise_methods, bfi10_methods, rfecv_methods)]

# Run combinations on HPC
for run, combination in enumerate(combinations):
    d = combination[0]
    m = combination[1]
    t = combination[2]
    noise = combination[3]
    bfi10 = combination[4]
    rfecv = combination[5]

    if rfecv:
        rfecv = '1'
    else:
        rfecv = '0'

    if noise:
        noise = '1'
    else:
        noise = '0'

    if bfi10:
        bfi10 = '1'
    else:
        bfi10 = '0'

    db_info = databases_info[d]

    # ------- Skip certain combinations -------
    # if db is not REMDE, only run one option of noise
    if db_info[3] == False and noise == '0':
        continue

    # if db is not NSC, only run one option of bfi10
    if db_info[4] == False and bfi10 == '0':
        continue

    # if f is embeddings, skip the rfe
    if db_info[5] == 'embeddings' and rfecv == '1':
        continue

    # ------- Create HPC config file -------

    # read the file into a list of lines
    lines = open(HPC_CONFIG, 'r').readlines()

    # now edit the last line of the list of lines
    addition = f' -b {b} --run {run} -d {d} -m {m} -t {t}  --leave-noisy {noise} --leave-bfi10 {bfi10} --timestamp {timestamp} --pca {pca} --rfecv {rfecv}'
    print(addition)
    new_last_line = (lines[-1].rstrip() + addition)
    lines[-1] = new_last_line

    filename = f"{FILE_DIR}/hpc_configs/sh/run-{timestamp}-{b}-{run}.sh"

    pythonfile = f'{FILE_DIR}/run.py'
    # now write the modified list back out to the file
    open(filename, 'w').writelines(lines)
    subprocess.call(['sbatch', filename, pythonfile]) # , shell=True