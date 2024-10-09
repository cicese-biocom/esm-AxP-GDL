#!/bin/bash

dataset="example/ExampleDatasetInference.csv"
pdb_path="example/ESMFold_pdbs/"
gdl_model_path="example/output/Checkpoints/epoch=2_train-loss=0.70_val-loss=0.70_(best-mcc).pt"
output_path="example/output/"
batch_size=512

python test.py \
    --dataset "$dataset" \
    --pdb_path "$pdb_path" \
    --gdl_model_path="$gdl_model_path" \
    --output_path="$output_path"  \
    --batch_size="$batch_size"
