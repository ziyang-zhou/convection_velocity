#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=0-2:00

source /project/m/moreaust/zzhou/pyenv/pyenv368_vtk/bin/activate
source /project/m/moreaust/zzhou/antares/v1.18.0/antares.env 
python extract_Rxt.py
