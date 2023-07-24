#!/bin/bash

source ~/.miniconda3/bin/activate
conda activate fairseq


# continual training (cif)
fairseq-hydra-train \
	task.data=/livingrooms/ray_chen/manifest/LS960 \
	common.wandb_project=fairseq \
	common.log_interval=50 \
	checkpoint.save_dir=base-continual-ofa-v1-inter \
	checkpoint.save_interval_updates=1000 \
	checkpoint.keep_interval_updates=30 \
	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/wav2vec2_base.pt \
	checkpoint.reset_optimizer=True \
	distributed_training.distributed_world_size=1 \
	+optimization.update_freq='[64]' \
	--config-dir examples/wav2vec/config/pretraining \
	--config-name wav2vec2_base_librispeech_ofa_v1 \
