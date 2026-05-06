import numpy as np
from pathlib import Path
from monai.data import FolderLayout
from monai.transforms import (
    EnsureTyped, EnsureChannelFirstd, Compose,
    CropForegroundd, LoadImaged, Orientationd, RandFlipd,
    RandCropByPosNegLabeld, ScaleIntensityRanged, Spacingd,
    RandRotate90d, ResizeWithPadOrCropd,
    Invertd, Activationsd, AsDiscreted, SaveImaged
)

def get_transforms(
    target_spacing: list,
    roi_size: list,
    a_min: float = -100.0,
    a_max: float = 400.0,
    b_min: float = 0.0,
    b_max: float = 1.0,
    flip_prob: float = 0.10,
    rotate_prob: float = 0.10
):
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Spacingd(keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=roi_size, 
            pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0, allow_smaller=True
        ),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=roi_size, mode='constant'),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=flip_prob),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=flip_prob),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=flip_prob),
        RandRotate90d(keys=["image", "label"], prob=rotate_prob, max_k=3),
        EnsureTyped(keys=["image", "label"])
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Spacingd(keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),
        EnsureTyped(keys=["image", "label"])
    ])
    
    return train_transforms, val_transforms

class PatientIdLayout(FolderLayout):
    """Writes <output_dir>/<patient_id>_seg.nii.gz, deriving patient_id
    from the parent folder of the source image (val_files[i]['image'].split('/')[-2])."""

    def __init__(self, output_dir, postfix="seg", extension=".nii.gz"):
        super().__init__(output_dir=output_dir, postfix=postfix, extension=extension)

    def filename(self, subject, **kwargs):
        # `subject` is the source image path passed in by SaveImaged via filename_or_obj
        pid = Path(subject).parent.name
        return Path(self.output_dir) / f"{pid}_{self.postfix}{self.ext}"

def get_post_transforms(val_transforms, output_dir):
    layout = PatientIdLayout(output_dir=output_dir, postfix="seg", extension=".nii.gz")

    save_pred = SaveImaged(
        keys="pred",
        meta_keys="pred_meta_dict",
        folder_layout=layout,
        writer="ITKWriter",
        resample=False,
        output_dtype=np.uint8,
        print_log=False,
    )
    save_pred.set_options(write_kwargs={"compression": True})

    return Compose([
        # orig_keys="image" allows Invertd to read image.applied_operations (the validation transforms)
        # and applies the inverse of each spatial operation to predictions
        Invertd( 
            keys="pred",
            transform=val_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ), # -> output is (C, H', W', D') with ' indicating inversions
        Activationsd(keys="pred", softmax=True), 
        # Activationsd applies softmax to logits to get probabilities along channel dim
        # now the output is (C, H', W', D') that sum to 1 along channel
        AsDiscreted(keys="pred", argmax=True), 
        # collapses C channels (2 in our case) to 1 channel of integer class indices through argmax
        # again, assigning each pixel the class index (either 0 or 1) based on the magnitude of the per-channel probability
        # output is (C, H', W', D') with per-pixel integer class assignment
        save_pred, # saves to disk
    ])
