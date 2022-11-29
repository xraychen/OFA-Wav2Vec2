import torch
import numpy as np
# import s3prl.hub as hub
import joblib
import yaml
import pandas as pd
from tqdm import tqdm
import os
import torchaudio

import argparse
from torch.utils.data import Dataset, DataLoader


# segment len penalty function
def pen(segment_length):
    return 1 - segment_length


# Simple implementation of dynamic programming based phoneme segmentation method given in
#   Towards unsupervised phone and word segmentation using self-supervised vector-quantized neural networks
#   (https://arxiv.org/abs/2012.07551, INTERSPEECH 2021)
# Author: Yuan Tseng (https://github.com/roger-tseng)
def segment(reps, kmeans_model, pen, lambd=35):
    """
    Inputs:
    reps        :   Representation sequence from self supervised model
    kmeans_model:   Pretrained scikit-learn MiniBatchKMeans model
    pen         :   penalty function penalizing segment length (longer segment, higher penalty)
    lambd       :   penalty weight (larger weight, longer segment)
    Outputs:
    boundaries  :   List of tokens at right boundaries of segments
                    (assuming token sequence starts from 1 to Tth token)
    label_token :   List of token labels for segments
    e.g. :
    If  tokens = [34, 55, 62, 83, 42]
        boundaries = [3, 5]
        label_token = [55, 83]
    then segmentation is :
    | 34 55 62 | 83 42 |
    |    55    |   83  |
    """
    # array of distances to closest cluster center, size: token sequence len * num of clusters
    distance_array = np.square(kmeans_model.transform(reps))
    alphas = [[0, None]]

    # Perform dynamic-programming-based segmentation
    for t in range(1, reps.shape[0] + 1):

        errors = []
        closest_centers = []
        for segment_length in range(1, t + 1):

            # array len = num of clusters
            # ith element is sum of distance from the last segment_length tokens until Tth token to the ith cluster center
            distance_subarray = distance_array[t - segment_length : t].sum(axis=0)

            closest_center = distance_subarray.argmin()
            error = (
                alphas[t - segment_length][0]
                + distance_subarray.min()
                + lambd * pen(segment_length)
            )

            closest_centers.append(closest_center)
            errors.append(error)

        errors = np.array(errors)
        alpha, a_min, closest = (
            errors.min(),
            t - 1 - errors.argmin(),
            closest_centers[errors.argmin()],
        )
        alphas.append([alpha, a_min, closest])

    # Backtrack to find optimal boundary tokens and label
    boundaries = []
    label_tokens = []
    tk = len(alphas) - 1
    while tk != 0:
        boundaries.append(tk)
        label_tokens.append(alphas[tk][2])
        tk = alphas[tk][1]
    boundaries.reverse()
    label_tokens.reverse()

    return boundaries, label_tokens


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config", type=str, default="pretrain/seq_distiller/config_runner.yaml"
    )
    parser.add_argument("--file_path", type=str)
    parser.add_argument(
        "--sets",
        type=str,
        default="train-clean-100,train-clean-360,train-other-500,dev-clean,dev-other,test-clean,test-other",
    )
    parser.add_argument("--check_load", action="store_true")

    parser.add_argument("--do_stage1", action="store_true")
    parser.add_argument("--stage1_dir", type=str, default=None)
    parser.add_argument("-s", "--feature", type=str, default="hidden_state_6")

    parser.add_argument("--do_stage2", action="store_true")
    parser.add_argument("--stage2_dir", type=str, default=None)
    parser.add_argument("--km_model", type=str, default="data/km_models/km100.bin")
    parser.add_argument("--lambd", type=float, default=35)
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--start_p", type=float, default=None)
    parser.add_argument("--end_p", type=float, default=None)

    return parser.parse_args()


def stage1(args, config, feat_paths):
    """Extract HuBERT token, use GPU to accelerate"""
    DEVICE = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    raise NotImplementedError
    model = getattr(hub, "hubert")().to(DEVICE)
    model.eval()

    print(f"Running STAGE 1 with DEIVICE: {DEVICE} ...")
    for feat_path in tqdm(feat_paths):
        out_filename = os.path.join(args.stage1_dir, feat_path.split(".")[0] + ".npy")

        if os.path.exists(out_filename):
            if args.check_load:
                try:
                    np.load(out_filename)
                    continue
                except ValueError as e:
                    print(e)
                    print("Load check failed, re-run this feat ...")
            else:
                continue

        wav, _ = torchaudio.load(os.path.join(config["libri_root"], feat_path))
        wav = wav.squeeze()  # (seq_len)
        reps = model([wav.to(DEVICE)])[args.feature].squeeze()
        reps = reps.cpu().detach().numpy()

        os.makedirs(
            os.path.join(
                args.stage1_dir, "/".join(feat_path.split(".")[0].split("/")[:-1])
            ),
            exist_ok=True,
        )
        out_filename = os.path.join(args.stage1_dir, feat_path.split(".")[0] + ".npy")
        np.save(out_filename, reps)


def stage2_sub(args, process_num, process_idx, feat_paths, kmeans_model, beta=1.0):

    my_feat_paths = feat_paths[process_idx::process_num]
    with tqdm(
        total=len(my_feat_paths),
        desc=f"worker {process_idx:02d}",
        dynamic_ncols=True,
        position=process_idx + 1,
    ) as pbar:
        for i, feat_path in enumerate(feat_paths[process_idx::process_num]):
            out_filename = os.path.join(
                args.stage2_dir, "temp", feat_path.split(".")[0] + ".pt"
            )

            if os.path.exists(out_filename):
                if args.check_load:
                    try:
                        torch.load(out_filename)
                        pbar.update(1)
                        continue
                    except EOFError as e:
                        print(e)
                        print("Load check failed, re-run this feat ...")
                else:
                    pbar.update(1)
                    continue

            reps = np.load(
                os.path.join(args.stage1_dir, feat_path.split(".")[0] + ".npy")
            )

            # Predict vector quantized tokens
            boundaries, label_tokens = segment(reps, kmeans_model, pen, args.lambd)

            ind_diff = torch.diff(
                torch.tensor(boundaries), prepend=torch.Tensor([0])
            ).long()

            output = torch.repeat_interleave(torch.tensor(label_tokens), ind_diff)

            assert reps.shape[0] == ind_diff.sum(
                -1
            ), f"{reps.shape[0]} != {ind_diff.sum(-1)}"
            assert boundaries[-1] == reps.shape[0]

            os.makedirs(
                os.path.join(
                    args.stage2_dir,
                    "temp",
                    "/".join(feat_path.split(".")[0].split("/")[:-1]),
                ),
                exist_ok=True,
            )

            out_filename = os.path.join(
                args.stage2_dir, "temp", feat_path.split(".")[0] + ".pt"
            )
            torch.save(torch.LongTensor(output), out_filename)

            pbar.update(1)

    return 0


def stage2(my_args, config, feat_paths):
    from multiprocessing import Pool, RLock

    # kmeans_model = joblib.load(config['km_model'])
    kmeans_model = joblib.load(my_args.km_model)
    pool = Pool(
        processes=my_args.num_workers, initargs=(RLock(),), initializer=tqdm.set_lock
    )
    jobs = [
        pool.apply_async(
            stage2_sub, args=(my_args, my_args.num_workers, i, feat_paths, kmeans_model)
        )
        for i in range(my_args.num_workers)
    ]

    pool.close()
    for job in jobs:
        job.get()

    print("\n" * (my_args.num_workers + 1))


def main():
    args = get_args()
    print(args)

    # with open(args.config, "r") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    #     config = config["pretrain_expert"]["datarc"]
    config = None

    # feat_paths
    # file_path = config["file_path"]
    file_path = args.file_path

    sets = args.sets.split(",")
    print(f"extracting {sets}")
    tables = [pd.read_csv(os.path.join(file_path, s + ".csv")) for s in sets]
    table = pd.concat(tables, ignore_index=True).sort_values(
        by=["length"], ascending=False
    )
    feat_paths = table["file_path"].tolist()

    # strip feat_paths
    print(f"Total number of audio is: {len(feat_paths)}")
    start_idx = int(args.start_p * len(feat_paths)) if args.start_p is not None else 0
    if args.end_p is not None:
        end_idx = int(args.end_p * len(feat_paths))
        feat_paths = feat_paths[start_idx:end_idx]
    else:
        end_idx = None
        feat_paths = feat_paths[start_idx:]

    print(f"Starting from index {start_idx} to index {end_idx} ...")

    if args.do_stage1:
        stage1(args, config, feat_paths)

    if args.do_stage2:
        stage2(args, config, feat_paths)


if __name__ == "__main__":
    main()
