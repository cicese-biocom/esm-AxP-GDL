#!/bin/bash

dataset="example/dataset/ExampleDataset.csv"
pdb_path="example/dataset/ESMFold_pdbs/"
gdl_model_path="example/output/Checkpoints/epoch=5_train-loss=0.88_val-loss=0.88.pt"
output_path="example/output/"
dropout_rate=0.25
batch_size=512

python inference.py \
    --dataset "$dataset" \
    --pdb_path "$pdb_path" \
    --gdl_model_path="$gdl_model_path" \
    --output_path="$output_path"  \
    --dropout_rate="$dropout_rate" \
    --batch_size="$batch_size"

