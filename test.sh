#!/bin/bash

dataset="example/ExampleDataset.csv"
pdb_path="example/ESMFold_pdbs/"
output_path="example/output/"
gdl_model_path="example/output/Checkpoints/gdl_model_name.pt"
feature_file_for_ad="example/output/Features/Features.csv"

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
