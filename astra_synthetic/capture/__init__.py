"""
ASTRA Synthetic Engine 7: Sampling / Capture / IQ-WAV File Generator.
Converts impaired complex baseband IQ into physical raw .iq, 2-channel .wav,
and SigMF capture files with exact format ground truth, quantization error metrics,
and separated visible/hidden metadata sidecars.
"""

from .models import CaptureRecord
from .generator import CaptureGenerator
from .layouts import format_iq_scalar_stream, deformat_iq_scalar_stream
from .quantization import apply_clipping, quantize_scalars, dequantize_scalars
from .raw_iq import write_raw_iq_file, read_raw_iq_file, resolve_numpy_dtype
from .wav_iq import write_wav_iq_file, read_wav_iq_file
from .sigmf_writer import write_sigmf_dataset, read_sigmf_iq_file
from .validators import validate_capture_record
from .serializers import save_capture_metadata_pair, write_dataset_manifest

__all__ = [
    "CaptureRecord",
    "CaptureGenerator",
    "format_iq_scalar_stream",
    "deformat_iq_scalar_stream",
    "apply_clipping",
    "quantize_scalars",
    "dequantize_scalars",
    "write_raw_iq_file",
    "read_raw_iq_file",
    "resolve_numpy_dtype",
    "write_wav_iq_file",
    "read_wav_iq_file",
    "write_sigmf_dataset",
    "read_sigmf_iq_file",
    "validate_capture_record",
    "save_capture_metadata_pair",
    "write_dataset_manifest",
]
