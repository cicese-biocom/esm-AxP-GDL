#!/bin/bash

dataset="example/dataset/ExampleDataset.csv"
pdb_path="example/dataset/ESMFold_pdbs/"
tertiary_structure_method='esmfold'
gdl_model_path="example/output/Checkpoints/epoch=5_train-loss=0.88_val-loss=0.88.pt"
output_path="example/output/"
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

