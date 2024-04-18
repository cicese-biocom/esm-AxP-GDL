#!/bin/bash
#SBATCH --job-name=esm-axp-gdl-env_TEST
#SBATCH --nodes=
#SBATCH --ntasks-per-node=
#SBATCH --output=slurm_esm-axp-gdl-env_TEST.out
#SBATCH --error=slurm_esm-axp-gdl-env_TEST.err
#SBATCH --time=00:00:00
#SBATCH --cluster=
#SBATCH --partition=
#SBATCH --gres=

module purge
module load python/ondemand-jupyter-python3.8
module load cuda/11.3.0

source activate esm-axp-gdl-env

cd /path/to/framework/directory

dataset="/path/to/the/input/CSV/file"
pdb_path="/path/to/the/directory/where/PDB/files/are/saved/or/loaded"
output_path="/path/to/save/the/predictions"
gdl_model_path="/path/to/the/model.pt"

tertiary_structure_method='esmfold'
dropout_rate=0.25
batch_size=512

python test.py \
    --dataset "$dataset" \
    --pdb_path "$pdb_path" \
    --gdl_model_path="$gdl_model_path" \
    --tertiary_structure_method="$tertiary_structure_method"  \
    --output_path="$output_path"  \
    --dropout_rate="$dropout_rate" \
    --batch_size="$batch_size"
