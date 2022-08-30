from argparse import ArgumentParser, Namespace
import os
import SimpleITK as sitk


def main(args: Namespace):
    root_dir = args.path
    for i in range(1, 251):
        id_ = f"sub-strokecase{i:04d}"
        print(f"id: {id_} done.")

        dwi_path = os.path.join(
            root_dir, f"rawdata/{id_}/ses-0001/{id_}_ses-0001_dwi.nii.gz"
        )
        fixed_image = sitk.ReadImage(dwi_path)

        flair_path = os.path.join(
            root_dir,
            f"rawdata/{id_}/ses-0001/{id_}_ses-0001_flair.nii.gz",
        )
        moving_image = sitk.ReadImage(flair_path)

        registered_image = sitk.Elastix(fixed_image, moving_image)
        flair_registered_path = os.path.join(
            root_dir,
            f"rawdata/{id_}/ses-0001/{id_}_ses-0001_flair_registered.nii.gz",
        )
        assert fixed_image.GetSize() == registered_image.GetSize()
        sitk.WriteImage(registered_image, flair_registered_path)


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="""FLAIR-DWI registration.""", add_help=False
    )
    parser.add_argument("path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
