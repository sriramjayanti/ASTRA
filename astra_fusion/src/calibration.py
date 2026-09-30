"""
ASTRA Post-Hoc Probability Calibration via Temperature Scaling.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TemperatureScaler:
    """
    Learns scalar temperature parameter T > 0 on validation set to calibrate softmax probabilities.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(max(1e-4, temperature))

    def scale_logits(self, logits: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Applies temperature scaling: z_cal = z / T
        """
        if isinstance(logits, torch.Tensor):
            return logits / self.temperature
        return (np.asarray(logits, dtype=np.float32) / self.temperature).astype(np.float32)

    def calibrate_probabilities(self, logits: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Scales logits and applies softmax: softmax(z / T)
        """
        if isinstance(logits, np.ndarray):
            z = torch.from_numpy(logits).float()
        else:
            z = logits.float()

        scaled = z / self.temperature
        probs = F.softmax(scaled, dim=-1).cpu().numpy()
        return probs

    def fit(
        self,
        val_logits: Union[np.ndarray, torch.Tensor],
        val_labels: Union[np.ndarray, torch.Tensor],
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Optimizes temperature T using Negative Log-Likelihood (NLL) on validation logits.
        """
        if isinstance(val_logits, np.ndarray):
            logits_t = torch.from_numpy(val_logits).float()
        else:
            logits_t = val_logits.float()

        if isinstance(val_labels, np.ndarray):
            labels_t = torch.from_numpy(val_labels).long()
        else:
            labels_t = val_labels.long()

        temp_param = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = optim.LBFGS([temp_param], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits_t / temp_param.clamp(min=1e-3)
            loss = criterion(scaled_logits, labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.temperature = float(temp_param.item())
        return self.temperature
