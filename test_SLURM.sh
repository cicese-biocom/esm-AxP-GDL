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
module load java/1.8.0_181-oracle

source activate esm-axp-gdl-env

cd /path/to/framework/directory

dataset="/path/to/the/input/CSV/file"
pdb_path="/path/to/the/directory/where/PDB/files/are/saved/or/loaded"
output_path="/path/to/save/the/predictions"
gdl_model_path="/path/to/the/model.pt"
feature_file_for_ad="/path/to/the/CSV/file/of/features/to/build/the/applicability/domain"

tertiary_structure_method='esmfold'
methods_for_ad='percentile_based(gc), percentile_based(perp), IF(gc_perp_aad), IF(perp_aad), IF(gc), IF(aad)'
batch_size=512
seed=0

python test.py \
    --dataset "$dataset" \
    --pdb_path "$pdb_path" \
    --gdl_model_path="$gdl_model_path" \
    --tertiary_structure_method="$tertiary_structure_method"  \
    --methods_for_ad="$methods_for_ad"  \
    --feature_file_for_ad="$feature_file_for_ad"  \
    --output_path="$output_path"  \
    --seed="$seed"
