#!/bin/bash

dataset="example/ExampleDataset.csv"
pdb_path="example/ESMFold_pdbs/"
tertiary_structure_method='esmfold'
gdl_model_path="example/output/Checkpoints/gdl_model_name.pt"
output_path="example/output/"
batch_size=512

python test.py \
    --dataset "$dataset" \
    --pdb_path "$pdb_path" \
    --gdl_model_path="$gdl_model_path" \
    --tertiary_structure_method="$tertiary_structure_method"  \
    --output_path="$output_path"  \
    --batch_size="$batch_size"

