import torch
from torch import nn
import timm


def create_vit_model(pretrained=True, model_name="vit_base_patch16_224", num_classes=1000):
    """
    Creates a ViT model for ImageNet classification.

    Args:
        pretrained (bool, optional): If True, loads pre-trained weights. Defaults to True.
        model_name (str, optional): Name of the pre-trained ViT model variant. Defaults to "vit_base_patch16_224".
        num_classes (int, optional): Number of output classes (defaults to 1000 for ImageNet). Defaults to 1000.

    Returns:
        nn.Module: The created ViT model.
    """

    model = timm.create_model(model_name, pretrained=pretrained)

    # Modify the final classification head
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    return model


# # Example usage

# # Use pre-trained model
# model = create_vit_model()

# # Use model with randomly initialized weights
# model = create_vit_model(pretrained=False)