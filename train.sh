#!/bin/bash

e=200
hd=128
d=10
esm2_representation="esm2_t36"
path_to_save_models="output_models/AMPDiscover/amp_esmt36_d10_hd128/"
log_file_name="TrainingLog_AMPDiscover"

python train.py \
    --dataset datasets/AMPDiscover/AMPDiscover.csv \
    --esm2_representation "$esm2_representation" \
    --tertiary_structure_method esmfold \
    --tertiary_structure_path datasets/AMPDiscover/ESMFold_pdbs/ \
    --e "$e" \
    --hd "$hd" \
    --path_to_save_models "$path_to_save_models" \
    --d "$d" \
    --log_file_name "$log_file_name"

