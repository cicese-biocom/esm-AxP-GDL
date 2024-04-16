#!/bin/bash

dataset="example/ExampleDataset.csv"
pdb_path="example/ESMFold_pdbs/"
tertiary_structure_method='esmfold'
gdl_model_path="example/output/"
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
number_of_epoch=5

python train.py \
    --dataset "$dataset" \
    --pdb_path "$pdb_path" \
    --tertiary_structure_method="$tertiary_structure_method"  \
    --gdl_model_path="$gdl_model_path"  \
    --esm2_representation "$esm2_representation" \
    --edge_construction_functions="$edge_construction_functions" \
    --distance_function="$distance_function" \
    --distance_threshold="$distance_threshold" \
    --amino_acid_representation="$amino_acid_representation" \
    --number_of_heads="$number_of_heads" \
    --hidden_layer_dimension="$hidden_layer_dimension" \
    --add_self_loops \
    --use_edge_attr \
    --learning_rate="$learning_rate" \
    --dropout_rate="$dropout_rate" \
    --batch_size="$batch_size" \
    --number_of_epoch="$number_of_epoch"

