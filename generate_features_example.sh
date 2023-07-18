#! /bin/bash

pos=example/positive
neg=example/negative

python generate_features.py -tr_ia3m $pos/a3m/ \
                            -tr_onpz $pos/npz/

python generate_features.py -tr_ia3m $neg/a3m/ \
                            -tr_onpz $neg/npz/


