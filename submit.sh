#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --partition=nice
#SBATCH --nodelist=hpc7-[62-65,01-04],hpc6-[06-08,32-33,35-40]
#SBATCH --mem=0
#SBATCH --time=10:00:00
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp


echo "Starting script"
source ~/.bashrc

echo "Loading conda"
conda activate kidneystone_seg

echo "Conda activated"
which python
python --version

echo "Launching script"
echo "Starting job on $SLURM_JOB_NODELIST"
srun --mpi=pmix_v3 python main_multinode.py
