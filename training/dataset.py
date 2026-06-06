import json
from pathlib import Path
from typing import Callable

import torch
from datasets import load_dataset, Dataset
from torch.utils.data import Dataset as TorchDataset

DATASET_NAME = "yashikota/birds-525-species-image-classification"
SPLITS = ("train", "validation", "test")


def load_split(split: str) -> Dataset:
    return load_dataset(DATASET_NAME, split=split)


def load_all_splits() -> dict[str, Dataset]:
    return {split: load_split(split) for split in SPLITS}


def get_num_classes(dataset: Dataset) -> int:
    return dataset.features["label"].num_classes


def get_label_names(dataset: Dataset) -> list[str]:
    return dataset.features["label"].names


def export_label_map(dataset: Dataset, output_path: str | Path = "labels.json") -> None:
    names = get_label_names(dataset)
    label_map = {str(i): name for i, name in enumerate(names)}
    Path(output_path).write_text(json.dumps(label_map, indent=2))


class BirdDataset(TorchDataset):
    def __init__(self, hf_dataset: Dataset, transform: Callable | None = None) -> None:
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        label = int(sample["label"])
        if self.transform:
            image = self.transform(image)
        return image, label

