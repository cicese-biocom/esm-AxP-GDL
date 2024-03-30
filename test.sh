#! /bin/bash

dataset="datasets/AMPDiscover/AMPDiscover.csv"
esm2_representation="esm2_t36",
pdb_path="datasets/AMPDiscover/ESMFold_pdbs/"
edge_construction_functions="distance_threshold"
distance_function="euclidean"
threshold=10
granularity="CA"
number_of_heads=8
hidden_layer_dimension=128
add_self_loops=True
use_edge_attr=False
dropout_rate=0.5
batch_size=512
model_path="output_models/AMPDiscover/amp_esmt36_d10_hd128/amp_esmt36_d10_hd128.pt"
log_filename="TrainingLog_AMPDiscover"
prediction_filename="PredictionLog_AMPDiscover"


python test.py \
    --dataset "$dataset" \
    --esm2_representation "$esm2_representation" \
    --pdb_path="$pdb_path"
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

