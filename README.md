# Factorizer for Stoke Lesion Segmentation

This repository contains the code and pre-trained models for our method based on [Factorizer](https://arxiv.org/abs/2202.12295) ([repo](https://github.com/pashtari/factorizer)) submitted to the [ISLES'22](https://isles22.grand-challenge.org/) challenge for the task of stroke lesion segmentation. The submitted model was an ensemble of Swin Factorizer and Res-U-Net (each described in the [paper](https://arxiv.org/abs/2202.12295)). This model ranked among the top-performing in the ISLES'22 test phase.


## Installation

1. Clone this repository:

```bash
$ git clone https://github.com/pashtari/factorizer-isles22.git
```

2. Install the dependencies:

``` bash
$ pip install -r requirements.txt --use-deprecated=legacy-resolver
```


## Data

Before training models, we first need to prepare the dataset by taking the following steps:

1. Register and download the official ISLES'22 dataset from [this link](https://isles22.grand-challenge.org/).

2. FLAIR-DWI co-registration by the following command:

```bash
$ python flair-dwi_registration.py dataset_dir
```

where `dataset_dir` must contain `rawdata` and `derivatives` folders. After running above, for each patient, the FLAIR image will be registered to the DWI space using [SimpleElastix](https://simpleelastix.github.io/), and a file named `{id}_ses-0001_flair_registered.nii.gz` will be made.

3. Download the [JSON file](dataset.json) and place it in the same folder as the dataset according to the following folders structure:

```bash
ISLES22
├── dataset.json # data properties
└── training # dataset_dir
    ├── rawdata
    │   ├── sub-strokecase0001
    │   └── ...
    └── derivatives
        ├── sub-strokecase0001
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


## Models

The Swin Factorizer and Res-U-Net models pre-trained on the ISLES22 dataset are provided in the following. The fold corresponding to each patient is provided in the [JSON file](dataset.json).

| Model           | #Params (M) | FLOPs (G) | Dice (%) | Fold | Config                                                          | Checkpoint                                                                                                                                      |
|-----------------|-------------|-----------|----------|------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| Res-U-Net       | 28.8        | 145       | 76.6     | 0    | [link](./configs/config_isles22_fold0_resunet.yaml)         | [link](./logs/fold0/resunet/version_0/checkpoints/epoch=1999-step=99999.ckpt) |
| Res-U-Net       | 28.8        | 145       | 80.2     | 1    | [link](./configs/config_isles22_fold1_resunet.yaml)         | [link](./logs/fold1/resunet/version_0/checkpoints/epoch=1999-step=99999.ckpt) |
| Res-U-Net       | 28.8        | 145       | 77.4     | 2    | [link](./configs/config_isles22_fold2_resunet.yaml)         | [link](./logs/fold2/resunet/version_0/checkpoints/epoch=1999-step=99999.ckpt) |
| Res-U-Net       | 28.8        | 145       | 81.2     | 3    | [link](./configs/config_isles22_fold3_resunet.yaml)         | [link](./logs/fold3/resunet/version_0/checkpoints/epoch=1999-step=99999.ckpt) |
| Res-U-Net       | 28.8        | 145       | 75.5     | 4    | [link](./configs/config_isles22_fold4_resunet.yaml)         | [link](./logs/fold4/resunet/version_0/checkpoints/epoch=1999-step=99999.ckpt) |
| Swin Factorizer | 7.3         | 29        | 76.2     | 0    | [link](./configs/config_isles22_fold0_swin-factorizer.yaml) | [link](./logs/fold0/swin-factorizer/version_0/checkpoints/epoch=1999-step=99999.ckpt) |
| Swin Factorizer | 7.3         | 29        | 78.9     | 1    | [link](./configs/config_isles22_fold1_swin-factorizer.yaml) | [link](./logs/fold1/swin-factorizer/version_0/checkpoints/epoch=1999-step=99999.ckpt) |
| Swin Factorizer | 7.3         | 29        | 78.5     | 2    | [link](./configs/config_isles22_fold2_swin-factorizer.yaml) | [link](./logs/fold2/swin-factorizer/version_0/checkpoints/epoch=1999-step=99999.ckpt) |
| Swin Factorizer | 7.3         | 29        | 80.6     | 3    | [link](./configs/config_isles22_fold3_swin-factorizer.yaml) | [link](./logs/fold3/swin-factorizer/version_0/checkpoints/epoch=1999-step=99999.ckpt) |
| Swin Factorizer | 7.3         | 29        | 78.6     | 4    | [link](./configs/config_isles22_fold4_swin-factorizer.yaml) | [link](./logs/fold4/swin-factorizer/version_0/checkpoints/epoch=1999-step=99999.ckpt) |


## Inference (CPU)

1. You need all the 10 pre-trained models in [`logs`](./logs) placed it inside the home directory, following is the folders structure below:

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
$ python predict.py --dwi ... --adc ... --flair ... --output ... 
```

If you want to use only some of the models rather than all of them, specify the argument `--checkpoints` with a list of paths to the model checkpoints.


## Docker


### Build Docker Image

1. Clone this repository:

```bash
$ git clone https://github.com/pashtari/factorizer-isles22.git
```

2. Build docker image:

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
