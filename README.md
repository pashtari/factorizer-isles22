# Factorizer for Stoke Lesion Segmentation

This repository contains the code and pre-trained models for our method based on [Factorizer](https://doi.org/10.1016/j.media.2022.102706) ([repo](https://github.com/pashtari/factorizer)) submitted to the [ISLES'22](https://isles22.grand-challenge.org/) challenge for the task of stroke lesion segmentation. The submitted model was an ensemble of Swin Factorizer and Res-U-Net (each described in the [paper](https://arxiv.org/abs/2202.12295)). This model ranked among top 3 in the ISLES'22 final leaderboard.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/pashtari/factorizer-isles22.git
cd factorizer-isles22
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Before training models, we first need to prepare the dataset by taking the following steps:

1. Register and download the official ISLES'22 dataset from [this link](https://isles22.grand-challenge.org/). The dataset directory have the following structure:

```bash
ISLES22
├── dataset_description.json
├── LICENSE
├── participants.tsv
├── README
├── derivatives
│   ├── sub-strokecase0001
│   ├── sub-strokecase0002
│   └── ...
├── sub-strokecase0001
├── sub-strokecase0002
└── ...
```

2. Perform FLAIR-DWI co-registration:

```bash
$ python flair-dwi_registration.py <dataset_dir>
```

where `<dataset_dir>` is the dataset directory. The script uses [SimpleElastix](https://simpleelastix.github.io/) to align FLAIR images to the DWI space, generating files like `{id}_ses-0001_flair_registered.nii.gz`.

3. Download [`datalist.json`](datalist.json) and place it in the dataset folder:

```bash
<dataset_dir>
├── datalist.json # <--
├── dataset_description.json
├── LICENSE
├── participants.tsv
├── README
├── derivatives
│   ├── sub-strokecase0001
│   ├── sub-strokecase0002
│   └── ...
├── sub-strokecase0001
├── sub-strokecase0002
└── ...
```


## Training

To train **Swin Factorizer** on the first fold of a 5-fold cross-validation on two GPUs, use the following command:

```bash
$ python train.py --config configs/config_isles22_fold0_swin-factorizer.yaml
```

where `config_isles22_fold0_swin-factorizer.yaml` is a config file. You can find all the config files in [./configs](./configs), but before using them, change their values of `data_properties` to yours. The model checkpoint will then be saved in `./logs/fold0/swin-factorizer/version_0/checkpoints` for this example.

Similarly, to train **Res-U-Net** on the first fold, use the following command:

```bash
$ python train.py --config configs/config_isles22_fold0_resunet.yaml
```


## Pre-Trained Models

The Swin Factorizer and Res-U-Net models pre-trained on the ISLES22 dataset are provided in the following. The fold corresponding to each patient is provided in the [JSON file](datalist.json).

| Model           | #Params (M) | FLOPs (G) | Dice (%) | Fold | Config                                                          | Checkpoint                                                                                                                                      |
|-----------------|-------------|-----------|----------|------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| Res-U-Net       | 28.8        | 145       | 76.6     | 0    | [link](./configs/config_isles22_fold0_resunet.yaml)         | [link](https://drive.google.com/file/d/11ta9QQgsCxFa6yhJ8P3NphWmTdOfV-Qh/view?usp=sharing) |
| Res-U-Net       | 28.8        | 145       | 80.2     | 1    | [link](./configs/config_isles22_fold1_resunet.yaml)         | [link](https://drive.google.com/file/d/129DwLsD4ADYA0gvLJ0aAamzMcnp3ztQb/view?usp=sharing) |
| Res-U-Net       | 28.8        | 145       | 77.4     | 2    | [link](./configs/config_isles22_fold2_resunet.yaml)         | [link](https://drive.google.com/file/d/12-bgW4W54W1-jLjUUAow9x5ZvyB5RhVj/view?usp=sharing) |
| Res-U-Net       | 28.8        | 145       | 81.2     | 3    | [link](./configs/config_isles22_fold3_resunet.yaml)         | [link](https://drive.google.com/file/d/11zuznPQvI9rUjW7XUR7wEianqoHvlflT/view?usp=sharing) |
| Res-U-Net       | 28.8        | 145       | 75.5     | 4    | [link](./configs/config_isles22_fold4_resunet.yaml)         | [link](https://drive.google.com/file/d/11zHYKqdzjfKcGDkkf2jqW5sr08QVQmf7/view?usp=sharing) |
| Swin Factorizer | 7.3         | 29        | 76.2     | 0    | [link](./configs/config_isles22_fold0_swin-factorizer.yaml) | [link](https://drive.google.com/file/d/11wFyWpCV9BqiHxIySef5Y1ZZoBEhfvvl/view?usp=sharing) |
| Swin Factorizer | 7.3         | 29        | 78.9     | 1    | [link](./configs/config_isles22_fold1_swin-factorizer.yaml) | [link](https://drive.google.com/file/d/127u43vv_-M9GJfT9Vrdas8AnNPg9MWNe/view?usp=sharing) |
| Swin Factorizer | 7.3         | 29        | 78.5     | 2    | [link](./configs/config_isles22_fold2_swin-factorizer.yaml) | [link](https://drive.google.com/file/d/1206o6L4fo15kqSrPgDwql0pxiFg7pz7S/view?usp=sharing) |
| Swin Factorizer | 7.3         | 29        | 80.6     | 3    | [link](./configs/config_isles22_fold3_swin-factorizer.yaml) | [link](https://drive.google.com/file/d/11zgQCa-Vgdnn5UbkCEVSKCIaWAhhxC1J/view?usp=sharing) |
| Swin Factorizer | 7.3         | 29        | 78.6     | 4    | [link](./configs/config_isles22_fold4_swin-factorizer.yaml) | [link](https://drive.google.com/file/d/11z_XnCIIgnZZYTQAaFL_Xq53oTnpY2BI/view?usp=sharing) |


## Inference (CPU)

1. Download all the 10 pre-trained models via the [Google Drive link](https://drive.google.com/drive/folders/1onYJehT1ecVj_ABP6j1NAQr_10OGwS8r?usp=sharing) as a folder named `logs` and place it inside the home directory, that is the structure of folders will be as follows:

```bash
factorizer-isles22
├── configs
│   ├── config_isles_fold0_swin-factorizer.yaml
│   └── ...
├── logs
│   ├── fold0
│   │   ├── swin-factorizer
│   │   │   └── version_0
│   │   │       ├── checkpoints
│   │   │       │   └── epoch=515-step=99999.ckpt
│   │   │       └── ...
│   │   └── ...
│   └── ...
├── train.py
├── predict.py
└── ...
```

2. To make predictions for a test case using the ensemble of all the 10 pre-trained models, run the following command:

```bash
$ python predict.py --dwi path/to/dwi --adc path/to/adc --flair path/to/flair --output <output_dir> 
```

If you want to use only some of the models rather than all of them, specify the argument `--checkpoints` with a list of paths to the model checkpoints.


## Docker


### Build Docker Image

1. Clone this repository:

```bash
$ git clone https://github.com/pashtari/factorizer-isles22.git
```

2. Download all the 10 pre-trained models via the [Google Drive link](https://drive.google.com/drive/folders/1onYJehT1ecVj_ABP6j1NAQr_10OGwS8r?usp=sharing) as a folder named `logs` and place it inside the home directory, that is the structure of folders will be as follows:

```bash
factorizer-isles22
├── configs
│   ├── config_isles_fold0_swin-factorizer.yaml
│   └── ...
├── logs
│   ├── fold0
│   │   ├── swin-factorizer
│   │   │   └── version_0
│   │   │       ├── checkpoints
│   │   │       │   └── epoch=515-step=99999.ckpt
│   │   │       └── ...
│   │   └── ...
│   └── ...
├── train.py
├── predict.py
└── ...
```


3. Build docker image:

```bash
$ docker build -t factorizer-isles22 ~ 
```


### Run Docker Image

For test images and their predictions organized into the following folder structure:

```bash
data
├── input
│   ├── sub-strokecase0001
│   │   ├── sub-strokecase0001_ses-0001_dwi.nii.gz
│   │   ├── sub-strokecase0001_ses-0001_adc.nii.gz
│   │   └── sub-strokecase0001_ses-0001_flair.nii.gz
│   └── ...
└── output
    └── sub-strokecase0001
        ├── sub-strokecase0001_ses-0001_pred.nii.gz
        └── ...
```

The inference can be performed as follows:

```bash
$ cd data
$ docker run -v ${PWD}/input:/input -v ${PWD}/output:/output factorizer-isles22 --dwi /input/sub-strokecase0001/sub-strokecase0001_ses-0001_dwi.nii.gz --adc /input/sub-strokecase0001/sub-strokecase0001_ses-0001_adc.nii.gz --flair /input/sub-strokecase0001/sub-strokecase0001_ses-0001_flair.nii.gz --output /output/sub-strokecase0001/sub-strokecase0001_ses-0001_pred.nii.gz
```


### Save Docker image:

```bash
$ docker save factorizer-isles22:latest | gzip > factorizer-isles22.tar.gz
```


## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


## Contact

This repo is currently maintained by Pooya Ashtari ([@pashtari](https://github.com/pashtari)).