from models.residual_unet import ResidualUNet

def get_model(config):
    """
    Model factory to instantiate models based on name and parameters.
    """
    model_name = config.get("model", {}).get("name", "residual_unet").lower()
    base_channels = config.get("model", {}).get("base_channels", 32)
    activation_type = config.get("model", {}).get("activation_type", "sigmoid")
    
    if model_name == "residual_unet":
        return ResidualUNet(
            base_channels=base_channels,
            activation_type=activation_type
        )
    elif model_name == "advanced_restoration_v1":
        from models.advanced_restoration import AdvancedRestorationv1
        return AdvancedRestorationv1(
            base_channels=base_channels,
            activation_type=activation_type
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
