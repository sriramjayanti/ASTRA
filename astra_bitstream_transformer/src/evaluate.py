"""
evaluate.py
Comprehensive evaluation metrics: Macro F1, per-class Precision/Recall/F1, boundary accuracy, and region IoU.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

from .models import StructureLabel, LABEL_NAMES, ID_TO_LABEL
from .postprocess import extract_contiguous_regions


def compute_sequence_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    ignore_index: int = -100,
    num_classes: int = 6
) -> Dict[str, Any]:
    """
    Compute sequence classification metrics on unpadded bit positions.
    """
    valid_mask = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    y_true = targets[valid_mask]
    y_pred = predictions[valid_mask]

    if len(y_true) == 0:
        return {"macro_f1": 0.0, "accuracy": 0.0, "per_class": {}}

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        zero_division=0
    )

    macro_f1 = float(np.mean(f1[sup > 0])) if np.any(sup > 0) else 0.0

    per_class_metrics = {}
    for c_id in range(num_classes):
        c_name = ID_TO_LABEL.get(c_id, f"CLASS_{c_id}")
        per_class_metrics[c_name] = {
            "precision": round(float(p[c_id]), 4),
            "recall": round(float(r[c_id]), 4),
            "f1": round(float(f1[c_id]), 4),
            "support": int(sup[c_id])
        }

    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "per_class": per_class_metrics,
        "confusion_matrix": conf_mat.tolist()
    }


def compute_boundary_f1(
    pred_boundaries: np.ndarray,
    true_boundaries: np.ndarray,
    tolerance: int = 4
) -> Dict[str, float]:
    """
    Evaluate transition boundary accuracy with +/- tolerance bits.
    """
    pred_idx = np.where(pred_boundaries > 0.5)[0]
    true_idx = np.where(true_boundaries > 0.5)[0]

    if len(true_idx) == 0 and len(pred_idx) == 0:
        return {"boundary_precision": 1.0, "boundary_recall": 1.0, "boundary_f1": 1.0}
    if len(true_idx) == 0 or len(pred_idx) == 0:
        return {"boundary_precision": 0.0, "boundary_recall": 0.0, "boundary_f1": 0.0}

    # Match predicted boundaries to true boundaries within tolerance
    tp = 0
    matched_true = set()
    for p in pred_idx:
        # Find closest true boundary
        diffs = np.abs(true_idx - p)
        min_d_idx = np.argmin(diffs)
        if diffs[min_d_idx] <= tolerance and min_d_idx not in matched_true:
            tp += 1
            matched_true.add(min_d_idx)

    prec = float(tp / max(1, len(pred_idx)))
    rec = float(len(matched_true) / max(1, len(true_idx)))
    f1 = float(2.0 * prec * rec / max(1e-6, prec + rec))

    return {
        "boundary_precision": round(prec, 4),
        "boundary_recall": round(rec, 4),
        "boundary_f1": round(f1, 4),
        "tolerance_bits": tolerance
    }


def compute_region_iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_label: int = StructureLabel.HEADER.value
) -> float:
    """Calculate Intersection-over-Union for a given region class."""
    true_mask = (y_true == target_label)
    pred_mask = (y_pred == target_label)

    intersection = np.sum(true_mask & pred_mask)
    union = np.sum(true_mask | pred_mask)

    if union == 0:
        return 1.0 if not np.any(true_mask) else 0.0

    return float(intersection / union)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu"),
    num_classes: int = 6
) -> Dict[str, Any]:
    """
    Run full evaluation loop on a validation dataloader.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_pred_bounds = []
    all_true_bounds = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["channels"].to(device)
            mask = batch["padding_mask"].to(device)
            targets = batch["labels"].numpy()
            b_targets = batch["boundary_targets"].numpy()

            out = model(x, padding_mask=mask)
            logits = out["sequence_logits"] # [B, L, C]
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            b_logits = out["boundary_logits"].cpu().numpy()
            pred_b = (1.0 / (1.0 + np.exp(-b_logits))) > 0.5

            all_preds.append(preds.reshape(-1))
            all_targets.append(targets.reshape(-1))
            all_pred_bounds.append(pred_b.reshape(-1))
            all_true_bounds.append(b_targets.reshape(-1))

    flat_preds = np.concatenate(all_preds)
    flat_targets = np.concatenate(all_targets)
    flat_pred_b = np.concatenate(all_pred_bounds)
    flat_true_b = np.concatenate(all_true_bounds)

    seq_metrics = compute_sequence_metrics(flat_preds, flat_targets, num_classes=num_classes)
    bound_metrics = compute_boundary_f1(flat_pred_b, flat_true_b, tolerance=4)

    header_iou = compute_region_iou(flat_targets, flat_preds, target_label=StructureLabel.HEADER.value)
    sync_iou = compute_region_iou(flat_targets, flat_preds, target_label=StructureLabel.SYNC.value)
    payload_iou = compute_region_iou(flat_targets, flat_preds, target_label=StructureLabel.PAYLOAD.value)

    combined_metrics = {
        **seq_metrics,
        **bound_metrics,
        "region_iou": {
            "SYNC": round(sync_iou, 4),
            "HEADER": round(header_iou, 4),
            "PAYLOAD": round(payload_iou, 4)
        }
    }

    return combined_metrics
