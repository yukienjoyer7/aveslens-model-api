import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.dataset import BirdDataset, get_label_names, get_num_classes, load_split
from training.model import build_model
from training.transforms import get_base_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_path: str | Path, num_classes: int) -> torch.nn.Module:
    model = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    return model.to(DEVICE)


def evaluate_per_class(
    checkpoint_path: str | Path = "checkpoints/best.pt",
    batch_size: int = 32,
    num_workers: int = 4,
    output_path: str | Path = "eval_report.json",
) -> None:
    test_hf = load_split("test")
    num_classes = get_num_classes(test_hf)
    label_names = get_label_names(test_hf)

    model = load_checkpoint(checkpoint_path, num_classes)
    model.eval()

    test_ds = BirdDataset(test_hf, transform=get_base_transforms())
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    correct = [0] * num_classes
    total = [0] * num_classes

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(dim=1)
            for pred, label in zip(preds, labels):
                total[label] += 1
                if pred == label:
                    correct[label] += 1

    overall_top1 = sum(correct) / sum(total)
    per_class = {
        label_names[i]: round(correct[i] / total[i], 4) if total[i] > 0 else None
        for i in range(num_classes)
    }

    report = {
        "overall_top1": round(overall_top1, 4),
        "num_classes": num_classes,
        "per_class_accuracy": per_class,
    }

    Path(output_path).write_text(json.dumps(report, indent=2))
    print(f"overall top-1: {overall_top1:.4f}")
    print(f"report saved → {output_path}")


if __name__ == "__main__":
    evaluate_per_class()
