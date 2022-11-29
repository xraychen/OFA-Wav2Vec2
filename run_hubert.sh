#!/bin/bash

source ~/.miniconda3/bin/activate
conda activate fairseq


# fairseq-hydra-train \
# 	task.data=/livingrooms/ray_chen/manifest/LS960 \
# 	task.label_dir=/livingrooms/ray_chen/labels \
# 	task.labels='["km"]' \
# 	model.label_rate=100 \
# 	common.wandb_project=hubert \
# 	checkpoint.save_dir=test \
# 	distributed_training.distributed_world_size=1 \
# 	+optimization.update_freq='[32]' \
# 	--config-dir examples/hubert/config/pretrain \
# 	--config-name hubert_base_librispeech

# # original training
# fairseq-hydra-train \
# 	task.data=/livingrooms/ray_chen/manifest/LS960 \
# 	task.label_dir=/livingrooms/ray_chen/l25 \
# 	task.labels='["km"]' \
# 	model.label_rate=50 \
# 	common.wandb_project=hubert \
# 	checkpoint.save_dir=hubert-l25 \
# 	distributed_training.distributed_world_size=1 \
# 	+optimization.update_freq='[32]' \
# 	--config-dir examples/hubert/config/pretrain \
# 	--config-name hubert_base_librispeech

# # original training
# fairseq-hydra-train \
# 	task.data=/livingrooms/ray_chen/manifest/LS960 \
# 	task.label_dir=/livingrooms/ray_chen/l25 \
# 	task.labels='["km"]' \
# 	model.label_rate=50 \
# 	common.wandb_project=hubert \
# 	checkpoint.save_dir=hubert-l25 \
# 	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/hubert_base_ls960.pt \
# 	checkpoint.reset_optimizer=True \
# 	distributed_training.distributed_world_size=1 \
# 	+optimization.update_freq='[32]' \
# 	--config-dir examples/hubert/config/pretrain \
# 	--config-name hubert_base_librispeech

# original training
fairseq-hydra-train \
	task.data=/livingrooms/ray_chen/manifest/LS960 \
	task.label_dir=/livingrooms/ray_chen/l25 \
	task.labels='["km"]' \
	model.label_rate=50 \
	common.wandb_project=hubert \
	checkpoint.save_dir=continual-l25 \
	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/hubert_base_ls960.pt \
	checkpoint.reset_optimizer=True \
	distributed_training.distributed_world_size=4 \
	distributed_training.nprocs_pre_node=6 \
	+optimization.update_freq='[8]' \
	--config-dir examples/hubert/config/pretrain \
	--config-name hubert_base_librispeech
