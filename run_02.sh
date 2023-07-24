#!/bin/bash

source ~/.miniconda3/bin/activate
conda activate fairseq


# # continual training (cif)
# WANDB_MODE=offline fairseq-hydra-train \
# 	task.data=/livingrooms/ray_chen/manifest/LS960 \
# 	common.wandb_project=fairseq \
# 	common.log_interval=10 \
# 	checkpoint.save_dir=debug \
# 	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/wav2vec_vox_new.pt \
# 	checkpoint.reset_optimizer=True \
# 	distributed_training.distributed_world_size=1 \
# 	+optimization.update_freq='[128]' \
# 	--config-dir examples/wav2vec/config/pretraining \
# 	--config-name wav2vec2_large_librispeech_cif_v1 \

# continual training (cif)
fairseq-hydra-train \
	task.data=/livingrooms/ray_chen/manifest/LS960 \
	common.wandb_project=fairseq \
	common.log_interval=50 \
	checkpoint.save_dir=large-continual-w2vu-v1 \
	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/wav2vec_vox_new.pt \
	checkpoint.reset_optimizer=True \
	distributed_training.distributed_world_size=1 \
	+optimization.update_freq='[128]' \
	--config-dir examples/wav2vec/config/pretraining \
	--config-name wav2vec2_large_librispeech_cif_v1 \
