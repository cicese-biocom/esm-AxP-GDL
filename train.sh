#! /bin/bash

pos=data/train_data/positive
neg=data/train_data/negative

mkdir -p $pos 
mkdir -p $neg

cp datasets/train\ datasets/XUAMP/XU_pretrain_train_positive.fasta $pos/XU_pretrain_train_positive.fasta
cp datasets/train\ datasets/XUAMP/XU_pretrain_train_negative.fasta $neg/XU_pretrain_train_negative.fasta
cp datasets/train\ datasets/XUAMP/XU_pretrain_val_positive.fasta $pos/XU_pretrain_val_positive.fasta
cp datasets/train\ datasets/XUAMP/XU_pretrain_val_negative.fasta $neg/XU_pretrain_val_negative.fasta
cp datasets/train\ datasets/XUAMP/XU_train_positive.fasta $pos/XU_train_positive.fasta
cp datasets/train\ datasets/XUAMP/XU_train_negative.fasta $neg/XU_train_negative.fasta
cp datasets/train\ datasets/XUAMP/XU_val_positive.fasta $pos/XU_val_positive.fasta
cp datasets/train\ datasets/XUAMP/XU_val_negative.fasta $neg/XU_val_negative.fasta

python generate_features.py -tr_ia3m $pos/a3m/ \
                            -tr_onpz $pos/npz/
python generate_features.py -tr_ia3m $neg/a3m/ \
                            -tr_onpz $neg/npz/
python generate_features.py -tr_ia3m $pos/a3m/ \
                            -tr_onpz $pos/npz/
python generate_features.py -tr_ia3m $neg/a3m/ \
                            -tr_onpz $neg/npz/

python train.py -pos_t $pos/XU_pretrain_train_positive.fasta \
                -pos_v $pos/XU_pretrain_val_positive.fasta \
                -pos_npz $pos/npz/ \
                -neg_t $neg/XU_pretrain_train_negative.fasta \
                -neg_v $neg/XU_pretrain_val_negative.fasta \
                -neg_npz $neg/npz/ \
                -save saved_models/XU.model

python train.py -pos_t $pos/XU_train_positive.fasta \
                -pos_v $pos/XU_val_positive.fasta \
                -pos_npz $pos/npz/ \
                -neg_t $neg/XU_train_negative.fasta \
                -neg_v $neg/XU_val_negative.fasta \
                -neg_npz $neg/npz/ \
                -lr 0.0001 -e 20 \
                -pretrained_model saved_models/auc_XU.model \
                -save saved_models/XU_final.model
