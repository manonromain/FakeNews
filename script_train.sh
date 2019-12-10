#!/bin/bash

for num_layers in 2
do
for model in  "GAT" "GraphSage" "GCN"
do
	for features in "all" "text_only" "user_only"
	do
        python train.py twitter16 --model_type $model --num_layers $num_layers --exp_name final16_${features}_${model} --features $features --standardize --num_epochs 500 --seed 43
done
done
done
