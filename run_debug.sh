#!/bin/bash

source ~/.miniconda3/bin/activate
conda activate fairseq


# continual training (cif)
fairseq-hydra-train \
	task.data=/livingrooms/ray_chen/manifest/LS960 \
	common.wandb_project=debug \
	checkpoint.save_dir=debug \
	checkpoint.restore_file=/home/ray_chen/fairseq-forked/outputs/2022-11-19/15-27-02/continual-cif-dpl25-v4/crash.pt \
	checkpoint.reset_optimizer=True \
	distributed_training.distributed_world_size=1 \
	+optimization.update_freq='[64]' \
	--config-dir examples/wav2vec/config/pretraining \
	--config-name wav2vec2_base_librispeech_cif_v2

# # continual training (cif)
# fairseq-hydra-train \
# 	task.data=/livingrooms/ray_chen/manifest/LS960 \
# 	common.wandb_project=fairseq \
# 	checkpoint.save_dir=continual-cif-dpl25-v6 \
# 	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/wav2vec2_base.pt \
# 	checkpoint.reset_optimizer=True \
# 	distributed_training.distributed_world_size=1 \
# 	+optimization.update_freq='[64]' \
# 	--config-dir examples/wav2vec/config/pretraining \
# 	--config-name wav2vec2_base_librispeech_cif_v2

# # continual training (cif)
# fairseq-hydra-train \
# 	task.data=/livingrooms/ray_chen/manifest/LS960 \
# 	common.wandb_project=fairseq \
# 	checkpoint.save_dir=continual-cif-dpl25-v3 \
# 	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/wav2vec2_base.pt \
# 	checkpoint.reset_optimizer=True \
# 	distributed_training.distributed_world_size=4 \
# 	+optimization.update_freq='[16]' \
# 	--config-dir examples/wav2vec/config/pretraining \
# 	--config-name wav2vec2_base_librispeech_cif_v2

# # continual training (cif)
# fairseq-hydra-train \
# 	task.data=/livingrooms/ray_chen/manifest/LS960 \
# 	common.wandb_project=fairseq \
# 	checkpoint.save_dir=continual-cif-dpl25-v5 \
# 	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/wav2vec2_base.pt \
# 	checkpoint.reset_optimizer=True \
# 	distributed_training.distributed_world_size=4 \
# 	+optimization.update_freq='[16]' \
# 	--config-dir examples/wav2vec/config/pretraining \
# 	--config-name wav2vec2_base_librispeech_cif_v2

# # continual training (cif)
# fairseq-hydra-train \
# 	task.data=/livingrooms/ray_chen/manifest/LS960 \
# 	common.wandb_project=fairseq \
# 	checkpoint.save_dir=debug \
# 	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/wav2vec2_base.pt \
# 	checkpoint.reset_optimizer=True \
# 	distributed_training.distributed_world_size=1 \
# 	+optimization.update_freq='[64]' \
# 	--config-dir examples/wav2vec/config/pretraining \
# 	--config-name wav2vec2_base_librispeech_cif

# # continual training (cif)
# fairseq-hydra-train \
# 	task.data=/livingrooms/ray_chen/manifest/LS960 \
# 	common.wandb_project=fairseq \
# 	checkpoint.save_dir=continual-cif-dpl25 \
# 	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/wav2vec2_base.pt \
# 	checkpoint.reset_optimizer=True \
# 	distributed_training.distributed_world_size=1 \
# 	+optimization.update_freq='[64]' \
# 	--config-dir examples/wav2vec/config/pretraining \
# 	--config-name wav2vec2_base_librispeech_cif

# # continual training (cif)
# fairseq-hydra-train \
# 	task.data=/livingrooms/ray_chen/manifest/LS960 \
# 	common.wandb_project=fairseq \
# 	checkpoint.save_dir=continual-cif-dpl25-v2 \
# 	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/wav2vec2_base.pt \
# 	checkpoint.reset_optimizer=True \
# 	distributed_training.distributed_world_size=1 \
# 	+optimization.update_freq='[64]' \
# 	--config-dir examples/wav2vec/config/pretraining \
# 	--config-name wav2vec2_base_librispeech_cif_v2

# # original training
# fairseq-hydra-train \
# 	task.data=/livingrooms/ray_chen/manifest/LS960 \
# 	common.wandb_project=fairseq \
# 	checkpoint.save_dir=origianl \
# 	distributed_training.distributed_world_size=4 \
# 	+optimization.update_freq='[16]' \
# 	--config-dir examples/wav2vec/config/pretraining \
# 	--config-name wav2vec2_base_librispeech

# # continual training
# fairseq-hydra-train \
# 	task.data=/livingrooms/ray_chen/manifest/LS960 \
# 	common.wandb_project=fairseq \
# 	checkpoint.save_dir=continual-avg4 \
# 	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/wav2vec2_base.pt \
# 	checkpoint.reset_optimizer=True \
# 	distributed_training.distributed_world_size=4 \
# 	+optimization.update_freq='[16]' \
# 	--config-dir examples/wav2vec/config/pretraining \
# 	--config-name wav2vec2_base_librispeech

# # continual training (fixed)
# fairseq-hydra-train \
# 	task.data=/livingrooms/ray_chen/manifest/LS960 \
# 	common.wandb_project=fairseq \
# 	checkpoint.save_dir=continual-avg4 \
# 	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/wav2vec2_base.pt \
# 	checkpoint.reset_optimizer=True \
# 	distributed_training.distributed_world_size=4 \
# 	+optimization.update_freq='[16]' \
# 	+model.subsample=fixed \
# 	+model.fixed_subsample_ratio=4 \
# 	--config-dir examples/wav2vec/config/pretraining \
# 	--config-name wav2vec2_base_librispeech

# # continual training (cif)
# fairseq-hydra-train \
# 	task.data=/livingrooms/ray_chen/manifest/LS960 \
# 	common.wandb_project=fairseq \
# 	checkpoint.save_dir=continual-cif \
# 	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/wav2vec2_base.pt \
# 	checkpoint.reset_optimizer=True \
# 	distributed_training.distributed_world_size=4 \
# 	+optimization.update_freq='[16]' \
# 	--config-dir examples/wav2vec/config/pretraining \
# 	--config-name wav2vec2_base_librispeech_cif
