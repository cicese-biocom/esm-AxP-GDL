#!/bin/bash
#SBATCH --job-name=esm-axp-gdl-env
#SBATCH --nodes=
#SBATCH --ntasks-per-node=
#SBATCH --output=slurm_esm-axp-gdl-env.out
#SBATCH --error=slurm_esm-axp-gdl-env.err
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
gdl_model_path="/path/to/the/directory/where/the/framework/output/is/saved"

tertiary_structure_method='esmfold'
esm2_representation="esm2_t33"
edge_construction_functions="distance_based_threshold"
distance_function="euclidean"
distance_threshold=10
amino_acid_representation="CA"
number_of_heads=8
hidden_layer_dimension=128
learning_rate=0.0001
dropout_rate=0.25
batch_size=512
number_of_epoch=200

python train.py \
    --dataset="$dataset" \
    --pdb_path="$pdb_path" \
    --tertiary_structure_method="$tertiary_structure_method"  \
    --gdl_model_path="$gdl_model_path"  \
    --esm2_representation="$esm2_representation" \
    --edge_construction_functions="$edge_construction_functions" \
    --distance_function="$distance_function" \
    --distance_threshold="$distance_threshold" \
    --amino_acid_representation="$amino_acid_representation" \
    --number_of_heads="$number_of_heads" \
    --hidden_layer_dimension="$hidden_layer_dimension" \
    --learning_rate="$learning_rate" \
    --dropout_rate="$dropout_rate" \
    --batch_size="$batch_size" \
    --number_of_epoch="$number_of_epoch" \
    --add_self_loops \
    --use_edge_attr \
    --save_ckpt_per_epoch
