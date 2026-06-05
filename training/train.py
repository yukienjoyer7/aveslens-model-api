from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from training.dataset import BirdDataset, load_split
from training.model import build_model, freeze_backbone, unfreeze_all
from training.transforms import get_base_transforms, get_train_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EpochMetrics:
    stage: str
    epoch: int
    train_loss: float
    val_loss: float
    top1: float
    top5: float

    def __str__(self) -> str:
        return (
            f"[{self.stage} {self.epoch}] "
            f"loss={self.train_loss:.4f} val_loss={self.val_loss:.4f} "
            f"top1={self.top1:.4f} top5={self.top5:.4f}"
        )


@dataclass
class TrainingHistory:
    epochs: list[EpochMetrics] = field(default_factory=list)

    def record(self, metrics: EpochMetrics) -> None:
        self.epochs.append(metrics)
        print(metrics)

    def best_top1(self) -> EpochMetrics:
        return max(self.epochs, key=lambda m: m.top1)


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"checkpoint saved → {path}")


def make_dataloaders(batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    train_ds = BirdDataset(load_split("train"), transform=get_train_transforms())
    val_ds = BirdDataset(load_split("validation"), transform=get_base_transforms())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def make_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int = 2):
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            total_loss += criterion(logits, labels).item()
            _, top5 = logits.topk(5, dim=1)
            correct_top1 += (top5[:, 0] == labels).sum().item()
            correct_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct_top1 / total, correct_top5 / total


def run_training(
    head_epochs: int = 5,
    full_epochs: int = 15,
    batch_size: int = 32,
    head_lr: float = 1e-4,
    backbone_lr: float = 1e-5,
    num_workers: int = 4,
    checkpoint_dir: str = "checkpoints",
) -> None:
    train_loader, val_loader = make_dataloaders(batch_size, num_workers)
    model = build_model(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    history = TrainingHistory()
    best_top1 = 0.0

    # stage 1: head only
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=head_lr)
    scheduler = make_scheduler(optimizer, total_epochs=head_epochs)
    for epoch in range(head_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, top1, top5 = evaluate(model, val_loader, criterion)
        scheduler.step()
        history.record(EpochMetrics("head", epoch + 1, train_loss, val_loss, top1, top5))
        if top1 > best_top1:
            best_top1 = top1
            save_checkpoint(model, Path(checkpoint_dir) / "best.pt")

    # stage 2: full finetune
    unfreeze_all(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=backbone_lr)
    scheduler = make_scheduler(optimizer, total_epochs=full_epochs)
    for epoch in range(full_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, top1, top5 = evaluate(model, val_loader, criterion)
        scheduler.step()
        history.record(EpochMetrics("full", epoch + 1, train_loss, val_loss, top1, top5))
        if top1 > best_top1:
            best_top1 = top1
            save_checkpoint(model, Path(checkpoint_dir) / "best.pt")

    print(f"\nbest epoch: {history.best_top1()}")


def run_from_config(config_path: str | Path = "training/config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    t = cfg["training"]
    run_training(
        head_epochs=t["head_epochs"],
        full_epochs=t["full_epochs"],
        batch_size=t["batch_size"],
        head_lr=t["head_lr"],
        backbone_lr=t["backbone_lr"],
        num_workers=t["num_workers"],
        checkpoint_dir=t["checkpoint_dir"],
    )


if __name__ == "__main__":
    run_from_config()
