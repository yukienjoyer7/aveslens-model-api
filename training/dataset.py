import json
from pathlib import Path

from datasets import load_dataset, Dataset

DATASET_NAME = "yashikota/birds-525-species-image-classification"
NUM_CLASSES = 525
SPLITS = ("train", "validation", "test")


def load_split(split: str) -> Dataset:
    return load_dataset(DATASET_NAME, split=split)


def load_all_splits() -> dict[str, Dataset]:
    return {split: load_split(split) for split in SPLITS}


def get_label_names(dataset: Dataset) -> list[str]:
    return dataset.features["label"].names


def export_label_map(dataset: Dataset, output_path: str | Path = "labels.json") -> None:
    names = get_label_names(dataset)
    label_map = {str(i): name for i, name in enumerate(names)}
    Path(output_path).write_text(json.dumps(label_map, indent=2))

