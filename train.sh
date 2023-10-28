#! /bin/bash

python train.py \
    -dataset datasets/DeepAVPpred/DeepAVPpred.csv \
    -esm2_representation esm2_t6 \
    -tertiary_structure_method esmfold \
    -tertiary_structure_path datasets/DeepAVPpred/ESMFold_pdbs/ \
    -tertiary_structure_operation_mode load \
    -e 20 \
    -b 512 \
    -hd 64 \
    -save output_models/ \
    -heads 8 \
    -d 20