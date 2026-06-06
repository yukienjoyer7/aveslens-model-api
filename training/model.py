import timm
import torch.nn as nn

MODEL_NAME = "swin_tiny_patch4_window7_224"


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("head"):
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True
