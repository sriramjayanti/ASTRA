"""
Inference API for ASTRA 2D Spectrogram Modulation Classifier.

Exposes standardized BranchPrediction structures for:
1. Single IQ window prediction: predict(iq_window)
2. Batch IQ prediction: predict_batch(iq_batch)
3. Top-K candidates and calibrated confidence scores
4. 256-D feature embeddings for downstream Fusion Engine
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn.functional as F

from .model import ASTRASpectrogramCNN
from .preprocessing import IQPreprocessor


class ASTRASpectrogramPredictor:
    """
    Production-ready Inference Engine for ASTRA 2D Spectrogram CNN.
    """
    def __init__(
        self,
        model: ASTRASpectrogramCNN,
        device: Optional[torch.device] = None,
        top_k: int = 3,
        preprocessor: Optional[IQPreprocessor] = None,
        model_name: str = "astra_spectrogram_cnn_v1",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.top_k = min(top_k, self.model.num_classes)
        self.preprocessor = preprocessor or IQPreprocessor()
        self.class_names = self.model.class_names
        self.model_name = model_name

    @torch.no_grad()
    def predict(
        self,
        iq_window: Union[np.ndarray, torch.Tensor],
        source_signal_id: Optional[str] = None,
        window_start: int = 0,
        return_embedding: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs inference on a single 1D complex IQ window [N] or [2, N].
        """
        if isinstance(iq_window, np.ndarray):
            clean_iq = self.preprocessor.process_numpy(iq_window)
            if np.iscomplexobj(clean_iq):
                i_ch = np.real(clean_iq).astype(np.float32)
                q_ch = np.imag(clean_iq).astype(np.float32)
                tensor_x = torch.from_numpy(np.stack([i_ch, q_ch], axis=0)).unsqueeze(0).to(self.device)
            else:
                tensor_x = torch.from_numpy(clean_iq).unsqueeze(0).to(self.device)
        else:
            tensor_x = iq_window.unsqueeze(0).to(self.device)

        logits = self.model(tensor_x)  # [1, num_classes]
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # [num_classes]
        raw_logits = logits.squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]
        conf = float(probs[pred_idx])

        # Top-K
        top_indices = np.argsort(probs)[::-1][: self.top_k]
        top_k_list = [
            {"class_name": self.class_names[i], "probability": float(probs[i]), "rank": r + 1}
            for r, i in enumerate(top_indices)
        ]

        prob_dict = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}

        result = {
            "model_name": self.model_name,
            "predicted_class": pred_class,
            "confidence": conf,
            "top_k": top_k_list,
            "probabilities": prob_dict,
            "logits": raw_logits.tolist(),
            "source_signal_id": source_signal_id,
            "window_start": window_start,
        }

        if return_embedding:
            embedding = self.model.extract_features(tensor_x).squeeze(0).cpu().numpy()
            result["feature_embedding"] = embedding.tolist()

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        iq_batch: Union[List[np.ndarray], torch.Tensor, np.ndarray],
        return_embedding: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Runs batch inference across multiple complex IQ windows.
        """
        if isinstance(iq_batch, list):
            return [self.predict(w, return_embedding=return_embedding) for w in iq_batch]

        if isinstance(iq_batch, np.ndarray):
            # Batch of complex signals [B, N]
            if np.iscomplexobj(iq_batch):
                i_ch = np.real(iq_batch).astype(np.float32)
                q_ch = np.imag(iq_batch).astype(np.float32)
                tensor_x = torch.from_numpy(np.stack([i_ch, q_ch], axis=1)).to(self.device)
            else:
                tensor_x = torch.from_numpy(iq_batch).to(self.device)
        else:
            tensor_x = iq_batch.to(self.device)

        logits = self.model(tensor_x)
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for b in range(len(probs)):
            p = probs[b]
            pred_idx = int(np.argmax(p))
            top_indices = np.argsort(p)[::-1][: self.top_k]
            top_k_list = [
                {"class_name": self.class_names[i], "probability": float(p[i]), "rank": r + 1}
                for r, i in enumerate(top_indices)
            ]
            prob_dict = {self.class_names[i]: float(p[i]) for i in range(len(self.class_names))}

            res = {
                "model_name": self.model_name,
                "predicted_class": self.class_names[pred_idx],
                "confidence": float(p[pred_idx]),
                "top_k": top_k_list,
                "probabilities": prob_dict,
                "logits": logits[b].cpu().numpy().tolist(),
            }
            results.append(res)

        return results
