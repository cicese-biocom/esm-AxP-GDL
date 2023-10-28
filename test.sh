#! /bin/bash

python test.py \
    -dataset datasets/DeepAVPpred/DeepAVPpred.csv \
    -esm2_representation esm2_t6 \
    -tertiary_structure_method esmfold \
    -tertiary_structure_path datasets/DeepAVPpred/ESMFold_pdbs/ \
    -tertiary_structure_operation_mode load \
    -b 512 \
    -save output_models/checkpoint_epoch20.pt \
    -drop 0.5 \
    -hd 64 \
    -heads 8 \
    -d 20
