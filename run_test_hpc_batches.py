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

# --- HPC PARAMETER GRID---
# # Overwrite default values
databases = ['spc-egemaps.pkl']
featuresets = ['e']
models = ['svm_rbf', 'rf', 'knn']
# models = ['svm_rbf']
traits = ['e', 'a', 'c', 'n', 'o']
# traits = ['e']
cal_methods = ['isotonic', 'sigmoid']

# Define with what HPC config the batches should be run
HPC_CONFIG = FILE_DIR + "/hpc_configs/run_short.sh"

for 
    # read the file into a list of lines
    lines = open(HPC_CONFIG, 'r').readlines()

    # now edit the last line of the list of lines
    addition = f' -b {b} -d {d} -f {f} -m {m} -t {t} --cal-method {c} -s {str(s)} -e {str(e)} --timestamp {timestamp}'
    print(addition)
    new_last_line = (lines[-1].rstrip() + addition)
    lines[-1] = new_last_line

    filename = f'{FILE_DIR}/hpc_configs/sh/run-{timestamp}-{b}-{d.replace(".","_")}-{f}-{m}-{t}-{str(s).replace(".","_")}-{str(e).replace(".","_")}-{c}.sh'

    pythonfile = f'{FILE_DIR}/run.py'
    # now write the modified list back out to the file
    open(filename, 'w').writelines(lines)
    subprocess.call(['sbatch', filename, pythonfile]) # , shell=True