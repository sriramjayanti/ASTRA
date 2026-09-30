"""
ASTRA Pipeline Stage Registry.
Central catalog of all 15 pipeline stages, descriptions, UI icons, and visualization metadata.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class StageMetadata:
    stage_id: int
    key: str
    name: str
    display_title: str
    icon_symbol: str
    domain: str
    description: str


class StageRegistry:
    """Provides canonical catalog of ASTRA Stages 1 to 15."""

    STAGES: List[StageMetadata] = [
        StageMetadata(1, "ingestion", "Signal Ingestion", "01. Signal Ingestion", "⚡", "RF", "Raw IQ ingestion, SNR & dynamic range measurement"),
        StageMetadata(2, "dsp", "DSP Preprocessing", "02. DSP Preprocessing", "〰", "DSP", "DC removal, energy normalization, bandpass filtering, STFT"),
        StageMetadata(3, "modulation", "Modulation Intelligence", "03. Modulation Discovery", "🧠", "AI/ML", "1D ResNet raw-IQ + 2D CNN Spectrogram neural fusion"),
        StageMetadata(4, "symbol_rate", "Symbol Rate Estimation", "04. Symbol Rate", "⏱", "DSP", "Cyclostationary & wave-difference baud rate candidates"),
        StageMetadata(5, "candidates", "Hypothesis Generation", "05. Candidate Universe", "🌌", "Hypothesis", "Multi-branch modulation × baud pipeline hypotheses"),
        StageMetadata(6, "synchronization", "Synchronization Engine", "06. Synchronization", "🎯", "Sync", "Gardner timing recovery & Costas loop CFO correction"),
        StageMetadata(7, "demodulation", "Demodulation Engine", "07. Demodulation", "⚄", "Baseband", "Constellation soft LLRs & hard symbol slicing"),
        StageMetadata(8, "interleaver", "Interleaver Analysis", "08. Deinterleaving", "🔀", "Channel", "Block, convolutional & helical permutation analysis"),
        StageMetadata(9, "fec", "FEC Error Correction", "09. FEC Decoder", "🛡", "Coding", "Soft-decision Viterbi, Reed-Solomon & LDPC correction"),
        StageMetadata(10, "validation", "Downstream Validation", "10. Validation Gates", "✔", "Integrity", "Deterministic CRC, syndrome & frame parity checks"),
        StageMetadata(11, "pipeline_scorer", "Pipeline Scorer", "11. Pipeline Ranker", "🏆", "ML Ranker", "XGBoost candidate competition & margin evaluation"),
        StageMetadata(12, "bitstream", "Bitstream Intelligence", "12. Bitstream Structure", "📊", "Bitstream", "Autocorrelation periodicity & frame-length discovery"),
        StageMetadata(13, "transformer", "Sequence Transformer", "13. CNN+Transformer", "🤖", "Deep Learning", "Neural sequence segmentation into Header, Payload, CRC"),
        StageMetadata(14, "payload", "Header/Payload Explorer", "14. Payload Explorer", "📦", "Protocol", "Frame boundary extraction, hex/ASCII decoding"),
        StageMetadata(15, "explainability", "Explainability Engine", "15. Reasoning Engine", "💡", "Explainability", "Evidence graph, field-specific canonical statuses"),
    ]

    _stage_map: Dict[int, StageMetadata] = {s.stage_id: s for s in STAGES}

    @classmethod
    def get_stage(cls, stage_id: int) -> Optional[StageMetadata]:
        return cls._stage_map.get(stage_id)

    @classmethod
    def get_all_stages(cls) -> List[StageMetadata]:
        return list(cls.STAGES)
