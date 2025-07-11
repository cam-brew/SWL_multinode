#!/bin/bash
## On how many hosts do we want to have the MPI program running?
## On each host, run only one task and use cpus-per-tasks number of cores of this host
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=nice
## Memory and walltime requirements
#SBATCH --mem-per-cpu=32G
#SBATCH --time=10:00:00

source ~/.bashrc
conda activate kidneystone_seg

VOLUME_ID=$1

hostfile=$(scontrol show hostnames $SLURM_JOB_NODELIST)
hosts = ($hostfile)
scheduler_node = ${hosts[0]}

echo "Starting Dask cluster ..."
srun --nodes=1 --ntasks=1 --nodelist=$scheduler_node --host $scheduler_node &

sleep 10

for host in "${hosts[@]}"; do
    srun --nodes=1 --ntasks=1 --cpus-per-task=48 --nodelist=$host \
        dask-worker $scheduler_node:8786 --nthreads=1 --nprocs=48 --memory-limit=0 &

done

wait 

python main_test.py --volume-id $VOLUME_ID --scheduler $scheduler_node:8786


