# Once-for-All Sequence Compression for Wav2Vec 2.0
This repository implements once-for-all sequence compression (ICASSP 2023) on Wav2Vec 2.0

- Paper title: Once-for-All Sequence Compression for Self-Supervised Speech Models
- Paper link: https://ieeexplore.ieee.org/abstract/document/10095025

## Enviromental Setup
This repo is modified from the original implementation of Wav2Vec 2.0 in the [fairseq toolkit](). To set up the environment, clone and install this repo with the following commands.
```bash
git clone https://github.com/xraychen/OFA-Wav2Vec2
cd OFA-Wav2Vec2 && pip install -e .
```

## Pre-train an OFA Wav2Vec 2.0
1. Prepare the training data manifest for LibriSpeech 960hr following the [instruction](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest).
```bash
pip install soundfile
python3 examples/wav2vec/wav2vec_manifest.py </path/to/LS960> --dest </path/to/manifest> --ext $ext --valid-percent $valid
```
2. Download the pre-extracted UASR segmentation of LibriSpeech 960hr from [here](https://www.dropbox.com/scl/fi/apwwgcftz6649e389tcrh/hubert_pseudo_alpha_w2vu.tar.gz?rlkey=ioz2hu867jom4m5pjtxjz1dja&dl=1) and you should get the following folder structure.
```
hubert_pseudo_alpha_w2vu
├── alpha
│   ├── train-clean-100
│   ├── train-clean-360
│   ├── train-other-500
│   └── dev-clean
└── boundaries
    ├── train-clean-100
    ├── train-clean-360
    ├── train-other-500
    └── dev-clean
```
3. Download the Wav2Vec 2.0 Base trained on 960hr Librispeech checkpoint from [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#pre-trained-models).
4. Run the following command to train an OFA sequence compression Wav2Vec 2.0 model.
```bash
fairseq-hydra-train \
    task.data=</path/to/manifest> \
    task.alpha_root=</path/to/alpha> \
    task.boundaries_root=</path/to/boundaries> \
    checkpoint.restore_file=</path/to/wav2vec2.pt> \
    checkpoint.reset_optimizer=True \
    checkpoint.save_dir=base-continual-ofa-v1 \
    checkpoint.save_interval_updates=1000 \
    checkpoint.keep_interval_updates=30 \
    distributed_training.distributed_world_size=1 \
    +optimization.update_freq='[64]' \
    --config-dir examples/wav2vec/config/pretraining \
    --config-name wav2vec2_base_librispeech_ofa_v1 \
```

You can find the pre-trained checkpoint here.
| Pre-trained Config    | Model Checkpoint | Sample Range of lambd | Traning Steps |
| -------- | ------- | -------- | ------- |
| [OFA 20-90ms](examples/wav2vec/config/pretraining/wav2vec2_base_librispeech_ofa_v1.yaml)  | [ckpt](https://www.dropbox.com/scl/fi/s7tzmz5h019dcg5seqlu0/checkpoint_582_5000.pt?rlkey=ymieopv1jyl1jgf5zubws8hv0&dl=1) | 0-1 | 5k |


## Extract Features from an OFA Wav2Vec 2.0
Here is the sample script to extract features from an OFA Wav2Vec 2.0, the value set to `model.overwrite_lambd` will change the sampling rate of the feature representation, please refer to our paper to find more details regards to relationship between lambd and sampling rate.
```python
import torch
from fairseq import checkpoint_utils


# load model from ckpt
ckpt = "/path/to/ofa_wav2vec2.pt"
models, _, _ = checkpoint_utils.load_model_ensemble_and_task([ckpt])
model = models[0].eval()

# overwrite the model's lambd value
model.overwrite_lambd = 0.7

# load waveform
wav_input_16khz = torch.ones(1, 16000 * 10)

# construct inputs
feats = wav_input_16khz.view(1, -1)
padding_mask = torch.BoolTensor(feats.shape).fill_(False)
inputs = {
    "source": feats,
    "padding_mask": padding_mask,
    "layer": 9,
}

# extract feature
with torch.no_grad():
    out = model.extract_features(**inputs)
    feat = out["x"]
    print(feat.size())
```

Use the extracted feature for any desired applications, and feel free to experiment with different `lambd` values!


## Citation
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
