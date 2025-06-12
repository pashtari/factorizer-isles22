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
| Res-U-Net       | 28.8        | 145       | 76.6     | 0    | [link](./configs/config_isles22_fold0_resunet.yaml)         | [link](https://www.dropbox.com/scl/fi/h9fx1dqxf6ste25bsbojl/epoch-1999-step-99999.ckpt?rlkey=kkn766oii8q6c3qi6lj55l6sb&st=2obnzmsx&dl=0) |
| Res-U-Net       | 28.8        | 145       | 80.2     | 1    | [link](./configs/config_isles22_fold1_resunet.yaml)         |  [link](https://www.dropbox.com/scl/fi/lwdh7oucsm5k0efg974o6/epoch-1999-step-99999.ckpt?rlkey=s8dceq3p04su150vee469467b&st=zsae7105&dl=0) |
| Res-U-Net       | 28.8        | 145       | 77.4     | 2    | [link](./configs/config_isles22_fold2_resunet.yaml)         | [link](https://www.dropbox.com/scl/fi/75lug085dw5v2qnoampul/epoch-1999-step-99999.ckpt?rlkey=dlnu62xw30nhodnstk280gojo&st=ns03wror&dl=0) |
| Res-U-Net       | 28.8        | 145       | 81.2     | 3    | [link](./configs/config_isles22_fold3_resunet.yaml)         | [link](https://www.dropbox.com/scl/fi/fp2zzbelf3ja86d26clkv/epoch-1999-step-99999.ckpt?rlkey=qk1uziqaf4ff8zpy7idf5m8fb&st=qzgh56xw&dl=0) |
| Res-U-Net       | 28.8        | 145       | 75.5     | 4    | [link](./configs/config_isles22_fold4_resunet.yaml)         | [link](https://www.dropbox.com/scl/fi/ciiotr1rsxnhvrfdzoj84/epoch-1999-step-99999.ckpt?rlkey=gwx89e2fpirwovvp7uv1w9ezo&st=3ruxh1u4&dl=0) |
| Swin Factorizer | 7.3         | 29        | 76.2     | 0    | [link](./configs/config_isles22_fold0_swin-factorizer.yaml) | [link](https://www.dropbox.com/scl/fi/xpbnrgjalqpt4sr48rewz/epoch-1999-step-99999.ckpt?rlkey=rydr0vqty88noj3n04z7iz12o&st=w1h8xbrh&dl=0) |
| Swin Factorizer | 7.3         | 29        | 78.9     | 1    | [link](./configs/config_isles22_fold1_swin-factorizer.yaml) | [link](https://www.dropbox.com/scl/fi/k55188gz3rgw474u7zga6/epoch-1999-step-99999.ckpt?rlkey=hnlhn5vedf3ppgb5vamv8hjx2&st=d9pt1azx&dl=0) |
| Swin Factorizer | 7.3         | 29        | 78.5     | 2    | [link](./configs/config_isles22_fold2_swin-factorizer.yaml) | [link](https://www.dropbox.com/scl/fi/lltlvpfdzbyfcbirrgi41/epoch-1999-step-99999.ckpt?rlkey=zrj6hr26l9ix6289ldguoyi39&st=mqcresk7&dl=0) |
| Swin Factorizer | 7.3         | 29        | 80.6     | 3    | [link](./configs/config_isles22_fold3_swin-factorizer.yaml) | [link](https://www.dropbox.com/scl/fi/lltlvpfdzbyfcbirrgi41/epoch-1999-step-99999.ckpt?rlkey=zrj6hr26l9ix6289ldguoyi39&st=mqcresk7&dl=0) |
| Swin Factorizer | 7.3         | 29        | 78.6     | 4    | [link](./configs/config_isles22_fold4_swin-factorizer.yaml) | [link](https://www.dropbox.com/scl/fi/ju5kid8zaxxrteldim6fd/epoch-1999-step-99999.ckpt?rlkey=3yzpdro40hvztf7thhpqc9x7g&st=f5ycwvui&dl=0) |


## Inference (CPU)

1. Download all the 10 pre-trained models via the [Dropbox link](https://www.dropbox.com/scl/fo/086lt5a0diobj7y81fdfv/APp_qpKjg2EQsgLxH6Sr8qo?rlkey=8lix61hp2vu6bq2g0fq4wyft0&st=6ft8bdd2&dl=0) as a folder named `logs` and place it inside the home directory, that is the structure of folders will be as follows:

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

2. Download all the 10 pre-trained models via the [Dropbox link](https://www.dropbox.com/scl/fo/086lt5a0diobj7y81fdfv/APp_qpKjg2EQsgLxH6Sr8qo?rlkey=8lix61hp2vu6bq2g0fq4wyft0&st=6ft8bdd2&dl=0) as a folder named `logs` and place it inside the home directory, that is the structure of folders will be as follows:

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