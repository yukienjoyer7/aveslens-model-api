from pathlib import Path

import torch

from training.dataset import get_num_classes, load_split
from training.model import build_model

DEVICE = torch.device("cpu")


def export_onnx(
    checkpoint_path: str | Path = "checkpoints/best.pt",
    output_path: str | Path = "checkpoints/model.onnx",
) -> None:
    test_hf = load_split("test")
    num_classes = get_num_classes(test_hf)

    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=14,
        dynamo=False,
    )

    print(f"exported → {output_path}  ({Path(output_path).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    export_onnx()
