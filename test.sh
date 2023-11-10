#! /bin/bash

hd=64
d=10
esm2_representation="esm2_t6"
trained_model="output_models/64_10_t6/64_10_t6_ckpt_1.pt"


python test.py \
    -dataset datasets/DeepAVPpred/DeepAVPpred.csv \
    --esm2_representation "$esm2_representation" \
    -tertiary_structure_method esmfold \
    -tertiary_structure_path datasets/DeepAVPpred/ESMFold_pdbs/ \
    --tertiary_structure_load_pdbs \
    -b 512 \
    -trained_model "$trained_model" \
    -drop 0.5 \
    --hd "$hd" \
    -heads 8 \
    --d "$d" \
    --add_self_loop





#!/bin/bash

e=2
hd=64
d=10
esm2_representation="esm2_t6"
path_to_save_models="output_models/64_10_t8/"

python train.py \
    --dataset datasets/AMPDiscover/AMPDiscover.csv \
    --esm2_representation "$esm2_representation" \
    --tertiary_structure_method esmfold \
    --tertiary_structure_path datasets/AMPDiscover/ESMFold_pdbs/ \
    --tertiary_structure_load_pdbs \
    --e "$e" \
    --hd "$hd" \
    --path_to_save_models "$path_to_save_models" \
    --d "$d" \
    --add_self_loop
