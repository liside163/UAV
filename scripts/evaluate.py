"""Evaluation entrypoint for UAV fault diagnosis models.

This script loads a trained checkpoint, builds the test dataset/dataloader based
on a YAML configuration, computes classification metrics, and visualizes the
confusion matrix.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from src.data.datasets import create_dataset
from src.data.transforms import build_augmentation
import src.data.transforms_signal  # noqa: F401  # ensure signal transforms are registered
import src.data.transforms_image  # noqa: F401  # ensure image transforms are registered
from src.evaluation import evaluate_model
from src.evaluation.visualization import plot_confusion_matrix
from src.features import create_feature_extractor
from src.models.backbones import create_backbone
from src.models.factory import TransferModel
from src.models.heads import create_head
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for evaluation.

    Returns:
        argparse.Namespace: Parsed arguments containing config path, checkpoint
        path, and optional device override.
    """

    parser = argparse.ArgumentParser(description="UAV fault diagnosis evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (e.g., best_model.pt)")
    parser.add_argument("--device", type=str, default=None, help="Device to use, e.g., 'cuda' or 'cpu'")
    return parser.parse_args()


def build_test_data_index(data_cfg: Dict[str, Any]) -> List[Tuple[str, int, Dict[str, Any]]]:
    """Construct the test split index.

    The configuration can directly provide ``test_index`` (a list of
    ``(path, label, meta)`` tuples). Alternatively, projects may expose a helper
    like ``build_test_data_index`` that reads ``cfg["data"]["test_split"]`` to
    generate the index (e.g., from CSV or directory scanning). Here we consume
    ``test_index`` when available and raise an error otherwise.

    Args:
        data_cfg (Dict[str, Any]): Data-related configuration.

    Returns:
        List[Tuple[str, int, Dict[str, Any]]]: Index entries for the test split.

    Raises:
        ValueError: If the configuration does not provide a test index.
    """

    test_index = data_cfg.get("test_index")
    if test_index is None:
        # Example: If you maintain test split files, replace this with
        # ``return build_test_data_index(data_cfg["test_split"])``.
        raise ValueError(
            "Data configuration must provide 'test_index' (list of (path, label, meta) tuples)"
        )
    return test_index


def create_test_dataloader(cfg: Dict[str, Any]) -> Tuple[DataLoader, List[str]]:
    """Create the test dataloader and class name list based on configuration.

    Args:
        cfg (Dict[str, Any]): Global configuration loaded from YAML/JSON.

    Returns:
        Tuple[DataLoader, List[str]]: A dataloader for the test set and the
        corresponding class names.
    """

    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset_name")
    if not dataset_name:
        raise ValueError("data.dataset_name is required in config")

    test_index = build_test_data_index(data_cfg)

    augmentation = build_augmentation(data_cfg.get("augmentation", []))
    feature_extractor = create_feature_extractor(
        data_cfg.get("feature_extractor"), **data_cfg.get("feature_extractor_params", {})
    )

    test_ds = create_dataset(
        dataset_name,
        data_index=test_index,
        transform=augmentation,
        feature_extractor=feature_extractor,
    )

    batch_size = data_cfg.get("test_batch_size") or cfg.get("train", {}).get("batch_size", 32)
    num_workers = data_cfg.get("num_workers") or cfg.get("train", {}).get("num_workers", os.cpu_count() or 2)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    class_names = data_cfg.get("class_names")
    if class_names is None:
        # Fallback to numeric labels aligned with confusion matrix dimensions.
        num_classes = data_cfg.get("num_classes") or len(set(idx[1] for idx in test_index))
        class_names = [str(i) for i in range(num_classes)]

    return test_loader, class_names


def create_model(cfg: Dict[str, Any]) -> TransferModel:
    """Construct the TransferModel from configuration for evaluation.

    Args:
        cfg (Dict[str, Any]): Global configuration.

    Returns:
        TransferModel: Initialized model ready for checkpoint loading.
    """

    model_cfg = cfg.get("model", {})
    backbone_cfg = model_cfg.get("backbone", {})
    head_cfg = model_cfg.get("head", {})

    if not backbone_cfg:
        raise ValueError("model.backbone configuration is required")
    if not head_cfg:
        raise ValueError("model.head configuration is required")

    backbone = create_backbone(**backbone_cfg)
    head = create_head(**head_cfg)
    freeze_backbone = model_cfg.get("freeze_backbone", False)
    return TransferModel(backbone, head, freeze_backbone=freeze_backbone)


def load_checkpoint(model: TransferModel, checkpoint_path: str, device: str) -> None:
    """Load model weights from a checkpoint file.

    Args:
        model (TransferModel): Model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.
        device (str): Target device for loading.
    """

    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get("model") if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()


def main() -> None:
    """Main evaluation routine."""

    args = parse_args()
    cfg = load_config(args.config)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    test_loader, class_names = create_test_dataloader(cfg)
    model = create_model(cfg)
    load_checkpoint(model, args.checkpoint, device)

    metrics = evaluate_model(model, test_loader, device)

    print("=== Evaluation Metrics ===")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Macro F1   : {metrics['macro_f1']:.4f}")
    print("Per-class precision:")
    for cls, prec in metrics["per_class_precision"].items():
        print(f"  Class {cls}: {prec:.4f}")
    print("Per-class recall:")
    for cls, rec in metrics["per_class_recall"].items():
        print(f"  Class {cls}: {rec:.4f}")

    conf_path = cfg.get("evaluation", {}).get("confusion_matrix_path", "outputs/confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, conf_path)
    print(f"Confusion matrix saved to: {conf_path}")


if __name__ == "__main__":
    main()
