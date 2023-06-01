#!/bin/sh
#you can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)
# The default partition is the 'general' partition
#SBATCH --partition=general
# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=short
# The default run (wall-clock) time is 1 minute
#SBATCH --time=3:59:59
# The default number of parallel tasks per job is 1
#SBATCH --ntasks=16
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=2
# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem-per-cpu=4GB
# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --mail-type=END

#link=https://storage.googleapis.com/example_download_data/subset_flac.tar
 
module use /opt/insy/modulefiles
module load miniconda
conda activate ser2
 
 
#the following deletes all the ipython profiles older than 1 day.
find ~/.ipython/profile_job* -maxdepth 0 -type d -ctime +1 | xargs rm -r
#create a new ipython profile appended with the job id number
profile=job_${SLURM_JOB_ID}
 
echo "Creating profile_${profile}"
~/.conda/envs/ser2/bin/ipython profile create ${profile}
 
~/.conda/envs/ser2/bin/ipcontroller --ip="*" --profile=${profile} &
sleep 30
 
#srun: runs ipengine on each available core
srun ~/.conda/envs/ser2/bin/ipengine --profile=${profile} --location=$(hostname) &
sleep 30
 
echo "Launching job for script $1"
~/.conda/envs/ser2/bin/python $1 -p ${profile} -b 136 -d spc-egemaps.pkl -f egemaps -m rf -t Extraversion --cal-method sigmoid -s 0 -e 0 --timestamp 230427_1604
