"""
ASTRA Modulation Intelligence V2 — Random Forest Family Support Integration.

Provides non-neural secondary evidence across the 4 primary signal families:
- FSK (supports 2-FSK, 4-FSK, MSK)
- PSK (supports BPSK, QPSK, 8PSK, DQPSK)
- QAM (supports 16QAM, 64QAM, 256QAM)
- UNKNOWN / Non-Target (supports noise, CW, chirps, anomalous signals)

Integrates smoothly into ASTRA Fusion Engine V2 to boost corresponding candidates
without hard-suppressing neural hypotheses.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from astra_modulation_v2.class_schema import MODULATION_CLASSES_V2, MODULATION_FAMILIES_V2


class RFFamilySupportAdapter:
    """Adapts Random Forest support model into probabilistic evidence for V2 fusion."""
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.root = Path(__file__).resolve().parent.parent
        self.checkpoint_path = checkpoint_path or str(self.root / "checkpoints" / "random_forest_support.joblib")
        self.engine = None
        self._load_engine()

    def _load_engine(self):
        try:
            from astra_random_forest.src.inference import RandomForestSupportEngine
            if os.path.exists(self.checkpoint_path):
                self.engine = RandomForestSupportEngine(checkpoint_path=self.checkpoint_path)
            else:
                self.engine = None
        except Exception as e:
            # Fallback if dependencies vary
            self.engine = None

    def get_family_probabilities(self, iq: np.ndarray, sample_rate: float = 192000.0) -> Dict[str, float]:
        """
        Returns probability distribution across broad signal families:
        {'FSK': p_fsk, 'PSK': p_psk, 'QAM': p_qam, 'UNKNOWN': p_unk}
        """
        default_uniform = {"FSK": 0.25, "PSK": 0.25, "QAM": 0.25, "UNKNOWN": 0.25}
        if self.engine is None or len(iq) == 0:
            return default_uniform
            
        try:
            # Extract DSP features and predict family probabilities
            features = self.engine.extract_features(iq, sample_rate=sample_rate)
            probs = self.engine.predict_family_proba(features)
            # Ensure all 4 families exist
            family_probs = {}
            for fam in ["FSK", "PSK", "QAM", "UNKNOWN"]:
                family_probs[fam] = float(probs.get(fam, 0.0))
                
            tot = sum(family_probs.values())
            if tot > 0:
                for fam in family_probs:
                    family_probs[fam] /= tot
            else:
                family_probs = default_uniform
                
            return family_probs
        except Exception:
            return default_uniform

    def get_class_support_vector(self, family_probs: Dict[str, float]) -> np.ndarray:
        """
        Maps broad family probabilities onto the canonical 11-class alphabet
        as a multiplicative or additive evidence vector [11].
        """
        support = np.zeros(len(MODULATION_CLASSES_V2), dtype=np.float32)
        for idx, cname in enumerate(MODULATION_CLASSES_V2):
            family = MODULATION_FAMILIES_V2.get(cname, "UNKNOWN")
            support[idx] = family_probs.get(family, 0.25)
        return support
