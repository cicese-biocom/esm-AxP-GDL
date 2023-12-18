#! /bin/bash

hd=128
d=10
esm2_representation="esm2_t36"
trained_model_path="output_models/AMPDiscover/amp_esmt36_d10_hd128/amp_esmt36_d10_hd128.pt"
log_file_name="TestLog_AMPDiscover"
test_result_file_name="TestResult_AMPDiscover"

python test.py \
    --dataset datasets/AMPDiscover/AMPDiscover.csv \
    --esm2_representation "$esm2_representation" \
    --tertiary_structure_method esmfold \
    --tertiary_structure_path datasets/AMPDiscover/ESMFold_pdbs/ \
    --b 512 \
    --trained_model_path "$trained_model_path" \
    --drop 0.5 \
    --hd "$hd" \
    --heads 8 \
    --d "$d" \
    --log_file_name "$log_file_name" \
    --test_result_file_name "$test_result_file_name"

