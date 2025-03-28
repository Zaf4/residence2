#!/bin/bash
#SBATCH -p normal
#SBATCH --account=ErbasGroup
#SBATCH --job-name=ZK
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 40
#SBATCH --time=47:59:59
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
##SBATCH --mail-type=ALL
##SBATCH --mail-user=zafer.kosar.physics@gmail.com

module unload gnu8
module unload openmpi3
module load intel
module load lammps

## EXPORT VALUES

export OMP_NUM_THREADS=1
export FI_PROVIDER=sockets
mpirun -np 40 lmp -in in.denge > log.txt
mpirun -np 40 lmp -in in.main > log2.txt
