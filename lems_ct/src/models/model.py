from monai.networks.nets import SegResNet

def get_segresnet(
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 2,
    init_filters: int = 16,
    blocks_down: list = [1, 2, 2, 4],
    blocks_up: list = [1, 1, 1]
) -> SegResNet:
    """
    Instantiates and returns the MONAI SegResNet model for LEMS-CT.
    
    Args:
        spatial_dims: Number of spatial dimensions (3 for 3D volumes).
        in_channels: Number of input channels.
        out_channels: Number of output channels (classes).
        init_filters: Initial number of filters.
        blocks_down: Number of downsampling blocks.
        blocks_up: Number of upsampling blocks.
        
    Returns:
        A MONAI SegResNet instance.
    """
    model = SegResNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,         
        blocks_down=blocks_down, 
        blocks_up=blocks_up,
    )
    
    return model