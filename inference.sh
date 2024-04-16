#!/bin/bash

dataset="example/ExampleDataset.csv"
pdb_path="example/ESMFold_pdbs/"
gdl_model_path="example/output/Checkpoints/gdl_model_name.pt"
output_path="example/output/"
dropout_rate=0.25
batch_size=512

python test.py \
    --dataset "$dataset" \
    --pdb_path "$pdb_path" \
    --gdl_model_path="$gdl_model_path" \
    --output_path="$output_path"  \
    --dropout_rate="$dropout_rate" \
    --batch_size="$batch_size"
