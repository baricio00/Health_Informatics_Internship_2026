from monai.transforms import (
    EnsureTyped, EnsureChannelFirstd, Compose,
    CropForegroundd, LoadImaged, Orientationd, RandFlipd,
    RandCropByPosNegLabeld, ScaleIntensityRanged, Spacingd,
    RandRotate90d, ResizeWithPadOrCropd
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