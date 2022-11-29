#!/bin/bash

source ~/.miniconda3/bin/activate
conda activate fairseq

cd examples/hubert/simple_kmeans

nshard=10
# for rank in $(seq 0 $((nshard - 1))); do
#     python3 dump_mfcc_feature.py /livingrooms/ray_chen/manifest/LS960 train ${nshard} ${rank} /livingrooms/ray_chen/feat
# done

# python3 learn_kmeans.py /livingrooms/ray_chen/feat train ${nshard} /livingrooms/ray_chen/feat/test.km 100 --percent 0.1

# for rank in $(seq 0 $((nshard - 1))); do
#     python3 dump_km_label.py /livingrooms/ray_chen/feat train /livingrooms/ray_chen/feat/test.km ${nshard} ${rank} /livingrooms/ray_chen/labels
# done

# for rank in $(seq 0 $((nshard - 1))); do
#     cat /livingrooms/ray_chen/labels/train_${rank}_${nshard}.km
# done > /livingrooms/ray_chen/labels/train.km

# for x in $(seq 0 $((100 - 1))); do
#     echo "$x 1"
# done >> /livingrooms/ray_chen/labels/dict.km.txt

for rank in $(seq 0 $((nshard - 1))); do
    python3 dump_mfcc_feature.py /livingrooms/ray_chen/manifest/LS960 valid ${nshard} ${rank} /livingrooms/ray_chen/feat
done

python3 learn_kmeans.py /livingrooms/ray_chen/feat valid ${nshard} /livingrooms/ray_chen/feat/test.km 100 --percent 0.1

for rank in $(seq 0 $((nshard - 1))); do
    python3 dump_km_label.py /livingrooms/ray_chen/feat valid /livingrooms/ray_chen/feat/test.km ${nshard} ${rank} /livingrooms/ray_chen/labels
done

for rank in $(seq 0 $((nshard - 1))); do
    cat /livingrooms/ray_chen/labels/valid_${rank}_${nshard}.km
done > /livingrooms/ray_chen/labels/valid.km
