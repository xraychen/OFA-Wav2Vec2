# OFA Sequence Compression on Wav2Vec 2.0

This is the Wav2Vec 2.0 version for the paper: [Once-for-All Sequence Compression for Self-Supervised Speech Models](https://ieeexplore.ieee.org/abstract/document/10095025), ICASSP 2023.

## Enviromental Setup
1. Install the fairseq toolkit from this repo.
2. Download pre-extracted UASR segmentation [here](https://www.dropbox.com/scl/fi/apwwgcftz6649e389tcrh/hubert_pseudo_alpha_w2vu.tar.gz?rlkey=ioz2hu867jom4m5pjtxjz1dja&dl=1) and set the path in the config file: [examples/wav2vec/config/pretraining/wav2vec2_base_librispeech_ofa_v1.yaml](examples/wav2vec/config/pretraining/wav2vec2_base_librispeech_ofa_v1.yaml).


## Pre-training
1. Prepare the pre-training data following the [instruction](examples/wav2vec/README.md).
2. Run the following command to pre-train OFA sequence compression Wav2Vec 2.0 model:
```
fairseq-hydra-train \
	task.data=/livingrooms/ray_chen/manifest/LS960 \
	checkpoint.save_dir=base-continual-ofa-v1-inter \
	checkpoint.save_interval_updates=1000 \
	checkpoint.keep_interval_updates=30 \
	checkpoint.restore_file=/livingrooms/ray_chen/fairseq-ckpt/wav2vec2_base.pt \
	checkpoint.reset_optimizer=True \
	distributed_training.distributed_world_size=1 \
	+optimization.update_freq='[64]' \
	--config-dir examples/wav2vec/config/pretraining \
	--config-name wav2vec2_base_librispeech_ofa_v1 \
```

You can find the pre-trained checkpoint here.
| Pre-trained Config    | Model Checkpoint |
| -------- | ------- |
| [OFA 20-90ms](examples/wav2vec/config/pretraining/wav2vec2_base_librispeech_ofa_v1.yaml)  | [Link](https://www.dropbox.com/scl/fi/s7tzmz5h019dcg5seqlu0/checkpoint_582_5000.pt?rlkey=ymieopv1jyl1jgf5zubws8hv0&dl=1) |


## Evaluation
Follow the instruction in the [repo](https://github.com/xraychen/OFA-Sequence-Compression) to evaluate on SUPERB downstream tasks.


To change compressing rate, simply specify the arg `--lambd`, please refers to the paper for more details. For example, the following command evaluate the model on Phone Recognition with `lambd=0.5`.
```
python3 run_downstream.py -n ExpName -m train -u wav2vec2_local -k /path/to/pre-train/ckpt -d ctc -c downstream/ctc/libriphone.yaml --lambd 0.5
```

## Ciatation
If you find the code helpful, please consider citing our paper!
```
@INPROCEEDINGS{10095025,
  author={Chen, Hsuan-Jui and Meng, Yen and Lee, Hung-yi},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Once-for-All Sequence Compression for Self-Supervised Speech Models},
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095025}}
```
