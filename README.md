# RayRoPE
[Project Page](https://rayrope.github.io/) | [arXiv](https://arxiv.org/abs/2601.15275v1)

This is the official code for paper: RayRoPE: Projective Ray Positional Encoding for Multi-view Attention.

## ToDos
- [] More detailed instruction on using thie repo coming soon.
- [] Instruction on donwloading datasets
- [] Release code for stereo depth experiments


## Setup

```
pip install -r requirements.txt 
pip install . # this will install two packages: prope, nvs
```

To make sure your setup works, you could run `pytest tests/`.

## Dataset

We first download the [RealEstate10K](https://google.github.io/realestate10k/) dataset using the script [`scripts/gen_imgs.py`](scripts/gen_imgs.py). Then we run [`scripts/gen_transforms.py`](scripts/gen_transforms.py) and [`scripts/data_processes.py`](scripts/data_processes.py) to convert the data into our data format.

Note we were not able to download all sequences as some of them are already invalid. We mark all sequences that we used for training and validation in the file [`assets/test_split_re10k.txt`](assets/test_split_re10k.txt) and [`assets/train_split_re10k.txt`](assets/train_split_re10k.txt) for reproducibility.

## Training

We support training with pixel-aligned camera conditioning (e.g., Plucker raymap, Naive raymap, Camray) or attention-based camera conditioning (e.g., GTA, PRoPE, RayRoPE) or the combination of them. For example training with `RayRoPE` on CO3D is via:

```
sbatch scripts/nvs_all.sh --dataset co3d --pos_enc d_pj+0_3d --depth_type predict_dsig

```

See `bash ./scripts/nvs_all.sh  -h` for helper information.


