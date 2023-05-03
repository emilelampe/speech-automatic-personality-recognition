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
featuresets = [config.f]
models = [config.m]
traits = [config.t]
start_cutoffs = [config.sc]
end_cutoffs = [config.ec]
cal_methods = [config.cal_method]
scoring_methods = [config.scoring]

# --- HPC GRID---
# # Overwrite default values
databases = ['spc-egemaps.pkl']
featuresets = ['e']
models = ['svm_rbf', 'rf', 'knn']
# models = ['svm_rbf']
traits = ['e', 'a', 'c', 'n', 'o']
# traits = ['e']
cal_methods = ['no_cal']
scoring = ['roc_auc', 'balanced_accuracy']
start_cutoffs = [0]
end_cutoffs = [0]

# Define with what HPC config the batches should be run
HPC_CONFIG = FILE_DIR + "/hpc_configs/run_short.sh"

# Create combinations of run parameters
combinations = [[d, f, m, t, s, e, c, scoring, timestamp, b] for d, f, m, t, s, e, c, scoring in itertools.product(databases, featuresets, models, traits, start_cutoffs, end_cutoffs, cal_methods, scoring_methods)]

# Run combinations on HPC
for run, combination in enumerate(combinations):
    d = combination[0]
    f = combination[1]
    m = combination[2]
    t = combination[3]
    s = combination[4]
    e = combination[5]
    c = combination[6]
    scoring = combination[7]
    # read the file into a list of lines
    lines = open(HPC_CONFIG, 'r').readlines()

    # now edit the last line of the list of lines
    addition = f' -b {b} --run {run} -d {d} -f {f} -m {m} -t {t} --cal-method {c} -s {str(s)} -e {str(e)} --timestamp {timestamp} --scoring {scoring}'
    print(addition)
    new_last_line = (lines[-1].rstrip() + addition)
    lines[-1] = new_last_line

    # filename = f'{FILE_DIR}/hpc_configs/sh/run-{timestamp}-{b}-{scoring}-{d.replace(".","_")}-{f}-{m}-{t}-{str(s).replace(".","_")}-{str(e).replace(".","_")}-{c}.sh'

    filename = f"{FILE_DIR}/hpc_configs/sh/run-{timestamp}-{b}-{run}.sh"

    pythonfile = f'{FILE_DIR}/run.py'
    # now write the modified list back out to the file
    open(filename, 'w').writelines(lines)
    subprocess.call(['sbatch', filename, pythonfile]) # , shell=True