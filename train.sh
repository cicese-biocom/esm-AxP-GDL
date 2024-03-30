#!/bin/bash

dataset="datasets/AMPDiscover/AMPDiscover.csv"
esm2_representation="esm2_t36",
tertiary_structure_method="esmfold"
edge_construction_functions="distance_threshold"
distance_function="euclidean"
threshold=10
granularity="CA"
number_of_heads=8
hidden_layer_dimension=128
add_self_loops=True
use_edge_attr=False
learning_rate=0.0001
dropout_rate=0.25
batch_size=512
number_of_epoch=200
model_path="output_models/AMPDiscover/amp_esmt36_d10_hd128/"
log_filename="TrainingLog_AMPDiscover"


python train.py \
    --dataset "$dataset" \
    --esm2_representation "$esm2_representation" \
    --tertiary_structure_method="$tertiary_structure_method"
    --edge_construction_functions="$edge_construction_functions"
    --distance_function="$distance_function"
    --threshold="$threshold"
    --granularity="$granularity"
    --number_of_heads="$number_of_heads"
    --hidden_layer_dimension="$hidden_layer_dimension"
    --add_self_loops="$add_self_loops"
    --use_edge_attr="$use_edge_attr"
    --learning_rate="$learning_rate"
    --dropout_rate="$dropout_rate"
    --batch_size="$batch_size"
    --number_of_epoch="$number_of_epoch"
    --model_path="$model_path"
    --log_filename="$log_filename"




