import timm
import torch.nn as nn

from training.dataset import NUM_CLASSES

MODEL_NAME = "swin_tiny_patch4_window7_224"


def build_backbone(pretrained: bool = True) -> nn.Module:
    return timm.create_model(MODEL_NAME, pretrained=pretrained)


def build_model(pretrained: bool = True) -> nn.Module:
    model = build_backbone(pretrained=pretrained)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, NUM_CLASSES)
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("head"):
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True
