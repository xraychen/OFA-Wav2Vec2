#!/bin/bash

source ~/.miniconda3/bin/activate
conda activate fairseq


# original training
fairseq-hydra-train \
	task.data=/livingrooms/ray_chen/manifest/LS960 \
	common.wandb_project=fairseq \
	checkpoint.save_dir=origianl \
	distributed_training.distributed_world_size=4 \
	+optimization.update_freq='[16]' \
	--config-dir examples/wav2vec/config/pretraining \
	--config-name wav2vec2_base_librispeech

# continual training
fairseq-hydra-train \
	task.data=/livingrooms/ray_chen/manifest/LS960 \
	common.wandb_project=fairseq \
	checkpoint.save_dir=continual-avg4 \
	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/wav2vec2_base.pt \
	checkpoint.reset_optimizer=True \
	distributed_training.distributed_world_size=4 \
	+optimization.update_freq='[16]' \
	--config-dir examples/wav2vec/config/pretraining \
	--config-name wav2vec2_base_librispeech

# continual training (fixed)
fairseq-hydra-train \
	task.data=/livingrooms/ray_chen/manifest/LS960 \
	common.wandb_project=fairseq \
	checkpoint.save_dir=continual-avg4 \
	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/wav2vec2_base.pt \
	checkpoint.reset_optimizer=True \
	distributed_training.distributed_world_size=4 \
	+optimization.update_freq='[16]' \
	+model.subsample=fixed \
	+model.fixed_subsample_ratio=4 \
	--config-dir examples/wav2vec/config/pretraining \
	--config-name wav2vec2_base_librispeech

# continual training (cif)
fairseq-hydra-train \
	task.data=/livingrooms/ray_chen/manifest/LS960 \
	common.wandb_project=fairseq \
	checkpoint.save_dir=continual-cif \
	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/wav2vec2_base.pt \
	checkpoint.reset_optimizer=True \
	distributed_training.distributed_world_size=4 \
	+optimization.update_freq='[16]' \
	--config-dir examples/wav2vec/config/pretraining \
	--config-name wav2vec2_base_librispeech_cif
