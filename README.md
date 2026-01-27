<p align="center">
  <img src="./assets/visual/rayrope_logo.png" height="80" />
  <img src="./assets/visual/rayrope_title.png" height="80" />
</p>

[Project Page](https://rayrope.github.io/) | [arXiv](https://arxiv.org/abs/2601.15275v1)

This is the official code for paper: RayRoPE: Projective Ray Positional Encoding for Multi-view Attention.

<p align="center">
  <img src="./assets/visual/method_fig_new.png" width="800" />
</p>

## Updates
- [01.26.2026] Update code on preparing the datasets (CO3D, RE10K, and Objaverse)
- [01.26.2026] Update instructions in this README.md

### ToDos
- Add instruction on the experiment options
- Release code for stereo depth experiments


## Environment

```
conda create -n rayrope python=3.11
pip install -r requirements.txt 
```

## Dataset

### CO3D
First download the CO3Dv2 from [here](https://github.com/facebookresearch/co3d/tree/main).

Then, pre-process the annotations: (code adapted from [RayDiffusion](https://github.com/jasonyzhang/RayDiffusion/blob/main/docs/train.md))
```
python ./scripts/co3d_preprocess --category all --precompute_bbox --co3d_v2_dir /path/to/co3d_v2
python ./scripts/co3d_preprocess --category all --co3d_v2_dir /path/to/co3d_v2
```

### RealEstate10K
There are two options to download RealEstate10K

#### Option 1
This is the option we use. Download the preprocessed RE10K dataset released by [pixelSplat](https://github.com/dcharatan/pixelsplat) with the following command:

```
wget -c "http://schadenfreude.csail.mit.edu:8000/re10k.zip" -P {YOUR_RE10K_DIR}/re10k_raw
```

Process the downloaded ray data using `scripts/lvsm_process_data_re10k.py`. This file is adapted from the officail LVSM repo.
```
python scripts/re10k_lvsm_process_data.py --base_path {YOUR_RE10K_DIR}/re10k_raw --output_dir {YOUR_RE10K_DIR}/re10k --mode ['train' or 'test']
```

The resulting data is organized in the format used by the official LVSM repo. However, our repo expects a slightly different format. We convert them using the following command:

```
python scripts/re10k_convert_data_format.py --source_dir {YOUR_RE10K_DIR}/re10k_raw --output_dir {YOUR_RE10K_DIR}/re10k_processed
```

#### Option 2
Download re10k directly following the instruction in the PRoPE repo. The resulting format should be compatible with our code. Note that some of the videos are no longer available on web.

### Objaverse
We use a 80K high-quality subset of Objaverse provided by [LGM](https://github.com/3DTopia/LGM?tab=readme-ov-file). The object ids are stored in `assets/kiuisobj_v1_merged_80K.csv'.

First, download the Objaverse 3D assets, following the instruction in the official [documentaion](https://objaverse.allenai.org/docs/intro). 

To better benchmark the ability of different positional encodings to capture large, diverse camera variations, we render the dataset with varying intrinsics, using `scripts/objv_render_vary_intrinsics.py`. For batch-rendering the dataset on a slurm cluster, we provide a submitit script `scripts/objv_submitit_batch_render.py`. To use, first set the path `OBJV_GLB_ROOT` and `OBJV_DIR` in the script, and run:
```
python scripts/objv_submitit_batch_render.py
```

## Training LVSM
Both training and testing can be launched by `scripts/nvs.sh`. Before using, first set the dataset paths at the beginning of the script. 
We support training LVSM with different multi-view positional encodings (e.g., Plucker raymap, GTA, PRoPE, RayRoPE). For example training with `RayRoPE` on CO3D is via:

```
scripts/nvs.sh --dataset co3d --pos_enc d_pj+0_3d --depth_type predict_dsig

```

See `bash ./scripts/nvs.sh  -h` for helper information.

## Acknowledgements
We build this code base on [PRoPE](https://github.com/liruilong940607/prope) and [Unimatch](https://github.com/autonomousvision/unimatch).


