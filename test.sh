#!/bin/bash

dataset="datasets/TestDataset/TestDataset.csv"
pdb_path="datasets/TestDataset/ESMFold_pdbs/"
tertiary_structure_method='esmfold'
gdl_model_path="output/TestDataset/Checkpoints/epoch=2_train-loss=0.68_val-loss=0.68.pt"
output_path="output/TestDataset/"
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

