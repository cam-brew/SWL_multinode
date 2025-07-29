#!/bin/bash
#SBATCH --nodes=6
#SBATCH --ntasks=6
#SBATCH --partition=nice
#SBATCH --nodelist=hpc7-[62-65,01-04],hpc6-[06-08,32-33,35-40],hpc3-[2101-2104,2401,2401-2402,2601-2603,2604,2701-2704,2801-2804,2901-2902],hpc5-[0501-0504,0601,0701-0703]
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --wait-all-nodes=1
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp


echo "Starting script"
source ~/.bashrc

echo "Loading conda"
conda activate ks_seg_dask

echo "Conda activated"
which python
python --version

echo "Launching script"
echo "Starting job on $SLURM_JOB_NODELIST"

srun --mpi=pmix_v3 python main_multinode.py "/home/esrf/cameron15a/Desktop/python/inputs/Real_05_01/seg_param_0007.txt"


echo "Completed python task and can be terminated"
