import os
import torch
import pandas as pd
from tqdm import tqdm

from multiprocessing import Pool, RLock


def subprocess(process_num, process_idx, all_files, labels_dir, output_dir):

    files = all_files[process_idx::process_num]

    with tqdm(
        total=len(files),
        desc=f"worker {process_idx:02d}",
        dynamic_ncols=True,
        position=process_idx + 1,
    ) as pbar:
        for file in files:
            filename = file.split(".")[0] + ".pt"

            label = os.path.join(labels_dir, "label_tokens", filename)
            index = os.path.join(labels_dir, "boundaries", filename)

            label = torch.load(label)
            index = torch.load(index)
            index = torch.diff(index, prepend=torch.Tensor([0])).long()

            output = torch.repeat_interleave(label, index)

            output_filename = os.path.join(output_dir, "temp", filename)
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            torch.save(output, output_filename)

            pbar.update(1)

    return 0


def main():
    DEVICE = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"DEVICE: {DEVICE}")

    split = "valid"

    tsv_file = f"/livingrooms/ray_chen/manifest/LS960/{split}.tsv"
    df = pd.read_csv(tsv_file, sep="\t", dtype=str, skiprows=[0], header=None)

    files = df.iloc[:, 0].tolist()

    num_workers = 36
    pool = Pool(processes=num_workers, initargs=(RLock(),), initializer=tqdm.set_lock)
    jobs = [
        pool.apply_async(
            subprocess,
            args=(
                num_workers,
                i,
                files,
                "/livingrooms/ray_chen/hubert_pseudo_alpha_l25",
                "/livingrooms/ray_chen/l25",
            ),
        )
        for i in range(num_workers)
    ]

    pool.close()
    for job in jobs:
        job.get()

    print("\n" * (num_workers + 1))

    output_file = f"/livingrooms/ray_chen/l25/{split}.km"

    with open(output_file, "w") as o:
        for i in tqdm(range(len(df))):
            filename = df.iloc[i, 0].split(".")[0] + ".pt"

            output = torch.load(
                os.path.join("/livingrooms/ray_chen/l25", "temp", filename)
            )

            # print("output", output)

            line = " ".join([str(int(e.item())) for e in output])

            o.write(line + "\n")


if __name__ == "__main__":
    main()
