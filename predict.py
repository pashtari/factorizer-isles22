import os
import glob
from argparse import ArgumentParser, Namespace

import numpy as np
import SimpleITK as sitk
import torch
from pytorch_lightning import seed_everything
from monai.transforms import SaveImaged

import factorizer as ft


seed_everything(42, workers=True)
torch.set_default_dtype(torch.float32)
cwd = os.path.dirname(__file__)


def main(args):
    # register flair to dwi
    fixed_image = sitk.ReadImage(args.dwi)
    moving_image = sitk.ReadImage(args.flair)
    registered_image = sitk.Elastix(fixed_image, moving_image)
    flair_file_name = os.path.basename(args.flair)
    flair_file_name = ".".join(
        [
            flair_file_name.split(".")[0] + "_registered",
            *flair_file_name.split(".")[1:],
        ]
    )
    registered_flair = os.path.join(
        os.path.dirname(args.flair), flair_file_name
    )
    sitk.WriteImage(registered_image, registered_flair)
    for file in glob.glob(cwd + "/TransformParameters.*"):
        os.remove(file)  # remove transform parameters files

    # init data module
    data_properties = {
        "test": [
            {
                "id": os.path.basename(args.output).split(".")[0],
                "image": [args.dwi, args.adc, registered_flair],
            }
        ]
    }
    dm = ft.ISLESDataModule(
        data_properties=data_properties,
        spacing=[2.0, 2.0, 2.0],
        spatial_size=[64, 64, 64],
        num_workers=0,
        cache_num=1,
        cache_rate=1.0,
        batch_size=1,
        seed=42,
    )
    dm.setup("test")

    # load (ensemble) model
    net_class = ft.Ensemble
    models = []
    for cp in args.checkpoints:
        models.append(ft.SemanticSegmentation.load_from_checkpoint(cp))

    num_models = len(args.checkpoints)
    weights = [1 / num_models for cp in args.checkpoints]
    net_params = {"models": models, "weights": weights}
    network = (net_class, net_params)
    inferer = ft.ISLESInferer(
        spacing=[2.0, 2.0, 2.0],
        spatial_size=[64, 64, 64],
        overlap=0.5,
        post="class",
    )
    model = ft.SemanticSegmentation(
        network=network, inferer=inferer, loss=(None,), metrics={}
    )

    # inference
    with torch.inference_mode():
        batch = [*dm.test_dataloader()][0]
        pred = model.inferer.get_postprocessed(batch, model)

    # remove registered flair
    os.remove(registered_flair)

    # save prediction
    pred = ft.decollate_batch(pred)[0]
    save = SaveImaged(
        "input",
        output_dir=os.path.dirname(args.output),
        output_dtype=np.uint8,
        output_postfix="",
        separate_folder=False,
        print_log=True,
    )
    pred["input"] = pred["input"].as_subclass(torch.Tensor) # convert to Tensor obj
    pred = ft.move_to(pred, device="cpu")
    save(pred)


def get_args() -> Namespace:
    default_checkpoints = [
        "logs/fold0/resunet/version_0/checkpoints/epoch=1999-step=99999.ckpt",
        "logs/fold1/resunet/version_0/checkpoints/epoch=1999-step=99999.ckpt",
        "logs/fold2/resunet/version_0/checkpoints/epoch=1999-step=99999.ckpt",
        "logs/fold3/resunet/version_0/checkpoints/epoch=1999-step=99999.ckpt",
        "logs/fold4/resunet/version_0/checkpoints/epoch=1999-step=99999.ckpt",
        "logs/fold0/swin-factorizer/version_0/checkpoints/epoch=1999-step=99999.ckpt",
        "logs/fold1/swin-factorizer/version_0/checkpoints/epoch=1999-step=99999.ckpt",
        "logs/fold2/swin-factorizer/version_0/checkpoints/epoch=1999-step=99999.ckpt",
        "logs/fold3/swin-factorizer/version_0/checkpoints/epoch=1999-step=99999.ckpt",
        "logs/fold4/swin-factorizer/version_0/checkpoints/epoch=1999-step=99999.ckpt",
    ]
    parser = ArgumentParser(
        description="""Do inference on test data.""", add_help=False
    )
    parser.add_argument(
        "--checkpoints", "-c", nargs="+", default=default_checkpoints
    )
    parser.add_argument("--dwi", "-d", required=True)
    parser.add_argument("--adc", "-a", required=True)
    parser.add_argument("--flair", "-f", required=True)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()
    return args


# class Args:
#     dwi = "/Users/pooya/Data/ISLES2022/training/rawdata/sub-strokecase0001/ses-0001/sub-strokecase0001_ses-0001_dwi.nii.gz"
#     adc = "/Users/pooya/Data/ISLES2022/training/rawdata/sub-strokecase0001/ses-0001/sub-strokecase0001_ses-0001_adc.nii.gz"
#     flair = "/Users/pooya/Data/ISLES2022/training/rawdata/sub-strokecase0001/ses-0001/sub-strokecase0001_ses-0001_flair.nii.gz"
#     output = "/Users/pooya/Data/ISLES2022/training/rawdata/sub-strokecase0001/ses-0001/sub-strokecase0001_ses-0001_pred.nii.gz"
#     checkpoints = [
#         "/Users/pooya/Library/CloudStorage/OneDrive-KULeuven/PhD_thesis/research/ISLES22/factorizer-isles22/logs/fold0/resunet/version_0/checkpoints/epoch=1999-step=99999.ckpt",
#         "/Users/pooya/Library/CloudStorage/OneDrive-KULeuven/PhD_thesis/research/ISLES22/factorizer-isles22/logs/fold0/swin-factorizer/version_0/checkpoints/epoch=1999-step=99999.ckpt",
#     ]


if __name__ == "__main__":
    args = get_args()
    # args = Args()
    main(args)
