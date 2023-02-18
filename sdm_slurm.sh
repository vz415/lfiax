#!/bin/bash

#SBATCH --job-name=sdm_equi        ## name of the job.
#SBATCH -A eehui_lab                      ## account to charge
#SBATCH -p standard                          ## partition/queue name
#SBATCH --error=./errors/error_%A_%a.txt    ## error log file name: %A is job id, %a is array task id
#SBATCH --output=./logs/out_%A_%a.out       ## output filename
#SBATCH --nodes=1                        ## number of nodes the job will use
#SBATCH --ntasks=1                       ## number of processes to launch for each array iteration
#SBATCH --cpus-per-task=1                ## number of cores the job needs
#SBATCH --time=10:00:00                   ## time limit for each array task
#SBATCH --array=53-102                      ## number of array tasks
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=vzaballa@uci.edu
                                ## $SLURM_ARRAY_TASK_ID takes values from 1 to 100 inclusive

## Activiating the conda environment
source ~/.bashrc

conda activate sbidoeman_2
#export LD_LIBRARY_PATH=/opt/apps/cuda/11.4.0/targets/x86_64-linux/lib/libcudart.so.11.0
export LD_LIBRARY_PATH=/data/homezvol1/vzaballa/.conda/envs/sbidoeman3/lib/

## Run the script
#python sbiDOEMAN/main_bma.py ++seed=$SLURM_ARRAY_TASK_ID ++num_design_rounds=5 ++BMP.model='onestep'
python sbiDOEMAN/main_random.py ++seed=$SLURM_ARRAY_TASK_ID ++SDM.random=True ++num_design_rounds=5 ++BMP.model="onestep"
#python sbiDOEMAN/main_equidistant.py ++seed=$SLURM_ARRAY_TASK_ID ++SDM.control=True 
