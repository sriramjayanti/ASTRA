import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn

from dataset import (
    load_config,
    complex_to_channels,
    IQPreprocessor,
    read_tim_file
)
from model import create_model

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ASTRA_INFERENCE")

# Global model cache to avoid reloading weights repeatedly in production/serving
_CACHED_MODEL = None
_CACHED_CONFIG = None
_CACHED_CLASSES = None
_CACHED_DEVICE = None
_CACHED_PATH = None
_CACHED_PREPROCESSOR = None


def load_inference_model(
    checkpoint_path: str = "checkpoints/best_model.pt",
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[nn.Module, Dict[str, Any], List[str], IQPreprocessor, torch.device]:
    global _CACHED_MODEL, _CACHED_CONFIG, _CACHED_CLASSES, _CACHED_DEVICE, _CACHED_PATH, _CACHED_PREPROCESSOR
    
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    if (
        _CACHED_MODEL is not None
        and _CACHED_PATH == str(ckpt_path)
        and _CACHED_DEVICE == target_device
    ):
        return _CACHED_MODEL, _CACHED_CONFIG, _CACHED_CLASSES, _CACHED_PREPROCESSOR, _CACHED_DEVICE

    ckpt = torch.load(ckpt_path, map_location=target_device, weights_only=False)
    config = ckpt.get("config", {})
    classes = ckpt.get("classes", config.get("classes", []))
    p_cfg = config.get("preprocessing", {})

    preprocessor = IQPreprocessor(
        remove_dc=p_cfg.get("remove_dc", True),
        normalization=p_cfg.get("normalization", "rms"),
        epsilon=p_cfg.get("epsilon", 1e-8)
    )

    model = create_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(target_device)
    model.eval()

    _CACHED_MODEL = model
    _CACHED_CONFIG = config
    _CACHED_CLASSES = classes
    _CACHED_PREPROCESSOR = preprocessor
    _CACHED_DEVICE = target_device
    _CACHED_PATH = str(ckpt_path)

    logger.info(f"Loaded ASTRA classifier on {target_device} from {ckpt_path}")
    return model, config, classes, preprocessor, target_device


def predict_modulation(
    iq_window: Union[np.ndarray, torch.Tensor],
    checkpoint_path: str = "checkpoints/best_model.pt",
    top_k: int = 3,
    preprocess: bool = True,
    model: Optional[nn.Module] = None,
    classes: Optional[List[str]] = None,
    preprocessor: Optional[IQPreprocessor] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """
    Inference API for ASTRA Modulation Classification.

    Input:
        iq_window: Raw complex IQ samples [N] or [2, N] or [N, 2]
        checkpoint_path: Path to best_model.pt (used if model is not pre-passed)
        top_k: Number of highest probability candidates to return
        preprocess: Whether to apply centralized DC removal and RMS normalization

    Returns:
        {
          "predicted_class": "qpsk",
          "confidence": 0.9123,
          "top_k": [
            {"class": "qpsk", "probability": 0.9123},
            {"class": "8psk", "probability": 0.0512},
            ...
          ],
          "probabilities": {
            "bpsk": 0.001,
            "qpsk": 0.9123,
            ...
          }
        }
    """
    if model is None:
        model, config, loaded_classes, loaded_prep, dev = load_inference_model(checkpoint_path, device=device)
        if classes is None:
            classes = loaded_classes
        if preprocessor is None:
            preprocessor = loaded_prep
    else:
        if device is None:
            dev = next(model.parameters()).device
        else:
            dev = torch.device(device)
        if classes is None:
            config = load_config()
            classes = config["classes"]
        if preprocessor is None:
            preprocessor = IQPreprocessor()

    # Format input tensor to shape [1, 2, N]
    if isinstance(iq_window, torch.Tensor):
        x = iq_window.detach().cpu().numpy()
    else:
        x = np.array(iq_window)

    # 1. Complex 1D array [N]
    if np.iscomplexobj(x):
        if preprocess:
            x = preprocessor.process(x)
        x_2ch = complex_to_channels(x)
    # 2. Real array with shape [N, 2] -> transpose to [2, N]
    elif x.ndim == 2 and x.shape[1] == 2 and x.shape[0] != 2:
        x_2ch = x.T.astype(np.float32)
        if preprocess:
            c = x_2ch[0] + 1j * x_2ch[1]
            c = preprocessor.process(c)
            x_2ch = complex_to_channels(c)
    # 3. Real array [2, N]
    elif x.ndim == 2 and x.shape[0] == 2:
        x_2ch = x.astype(np.float32)
        if preprocess:
            c = x_2ch[0] + 1j * x_2ch[1]
            c = preprocessor.process(c)
            x_2ch = complex_to_channels(c)
    else:
        raise ValueError(f"Unsupported IQ window shape or dtype: shape={x.shape}, dtype={x.dtype}")

    # Add batch dimension -> [1, 2, N]
    input_tensor = torch.from_numpy(x_2ch).unsqueeze(0).float().to(dev)

    with torch.no_grad():
        logits = model(input_tensor)
        probs_t = torch.softmax(logits, dim=1).squeeze(0)
        probs = probs_t.cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    prob_dict = {cls_name: float(round(float(probs[i]), 6)) for i, cls_name in enumerate(classes)}

    top_indices = np.argsort(probs)[::-1][:top_k]
    top_k_list = [
        {"class": classes[idx], "probability": float(round(float(probs[idx]), 6))}
        for idx in top_indices
    ]

    return {
        "predicted_class": pred_class,
        "confidence": float(round(confidence, 6)),
        "top_k": top_k_list,
        "probabilities": prob_dict
    }


if __name__ == "__main__":
    dummy_iq = np.random.randn(2048) + 1j * np.random.randn(2048)
    print("Testing inference module with synthetic test sample...")
    
    ckpt_test = Path("checkpoints/best_model.pt")
    if ckpt_test.exists():
        res = predict_modulation(dummy_iq, checkpoint_path=str(ckpt_test))
        print(json.dumps(res, indent=2))
    else:
        print("Checkpoint not found. Testing with randomly initialized model.")
        m = create_model()
        res = predict_modulation(dummy_iq, model=m)
        print(json.dumps(res, indent=2))
