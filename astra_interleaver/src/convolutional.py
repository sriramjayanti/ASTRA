"""
convolutional.py
Stateful Ramsey/Forney convolutional interleaver and deinterleaver engine.
Supports branch count, delay steps, latency compensation, and persistent state across chunks.
"""

from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np
from .models import ConvolutionalDeinterleaverState, InterleaverFamily


class ConvolutionalInterleaverEngine:
    """
    Transmitter Convolutional Interleaver (Forney type).
    Branch j (0 <= j < B) has delay line of j * M registers.
    """
    def __init__(self, branch_count: int, delay_step: int):
        self.branch_count = int(branch_count)
        self.delay_step = int(delay_step)
        self.registers: List[List[float]] = []
        self.reset()
        
    def reset(self):
        self.registers = []
        for j in range(self.branch_count):
            delay_len = j * self.delay_step
            self.registers.append([0.0] * delay_len)
        self.pos = 0

    def process_symbol(self, val: float) -> float:
        branch = self.pos
        self.pos = (self.pos + 1) % self.branch_count
        reg = self.registers[branch]
        if len(reg) == 0:
            return float(val)
        out_val = reg.pop(0)
        reg.append(float(val))
        return out_val

    def process_array(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        out = np.zeros(len(arr), dtype=arr.dtype)
        for i, val in enumerate(arr):
            out[i] = self.process_symbol(float(val))
        return out


class ConvolutionalDeinterleaverEngine:
    """
    Receiver Convolutional Deinterleaver (Forney type).
    Branch j (0 <= j < B) has delay line of (B - 1 - j) * M registers.
    Total system delay = B * (B - 1) * M samples.
    """
    def __init__(self, branch_count: int, delay_step: int, state: Optional[ConvolutionalDeinterleaverState] = None):
        self.branch_count = int(branch_count)
        self.delay_step = int(delay_step)
        if state is not None:
            self.state = state
        else:
            self.state = ConvolutionalDeinterleaverState(
                branch_count=self.branch_count,
                delay_step=self.delay_step
            )

    @property
    def latency_bits(self) -> int:
        """Total bit delay through interleaver + deinterleaver pair."""
        return self.branch_count * (self.branch_count - 1) * self.delay_step

    def process_symbol(self, val: float) -> float:
        branch = self.state.commutator_pos
        self.state.commutator_pos = (self.state.commutator_pos + 1) % self.branch_count
        self.state.total_processed += 1
        reg = self.state.shift_registers[branch]
        if len(reg) == 0:
            return float(val)
        out_val = reg.pop(0)
        reg.append(float(val))
        return float(out_val)

    def process_array(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        out = np.zeros(len(arr), dtype=arr.dtype)
        for i, val in enumerate(arr):
            out[i] = self.process_symbol(float(val))
        return out


def interleave_convolutional(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray],
    branch_count: int,
    delay_step: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Interleaves bitstream and optional soft LLRs with a Forney convolutional interleaver.
    """
    engine_hard = ConvolutionalInterleaverEngine(branch_count, delay_step)
    interleaved_hard = engine_hard.process_array(hard_bits).astype(np.uint8)
    
    interleaved_soft = None
    if soft_llrs is not None:
        engine_soft = ConvolutionalInterleaverEngine(branch_count, delay_step)
        interleaved_soft = engine_soft.process_array(soft_llrs).astype(np.float32)
        
    return interleaved_hard, interleaved_soft


def deinterleave_convolutional(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray],
    branch_count: int,
    delay_step: int,
    state: Optional[ConvolutionalDeinterleaverState] = None,
    trim_latency: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], ConvolutionalDeinterleaverState, int]:
    """
    Deinterleaves bitstream and optional soft LLRs with a Forney convolutional deinterleaver.
    
    Args:
        hard_bits: 1D uint8 array
        soft_llrs: Optional 1D float32 array
        branch_count: Number of branches B
        delay_step: Delay step multiplier M
        state: Optional persistent state for streaming chunks
        trim_latency: If True, trims the startup latency bits from the output head.
        
    Returns:
        (deinterleaved_hard, deinterleaved_soft, updated_state, latency_bits)
    """
    if branch_count <= 0 or delay_step <= 0:
        raise ValueError(f"Convolutional branch_count ({branch_count}) and delay_step ({delay_step}) must be positive")
        
    engine_hard = ConvolutionalDeinterleaverEngine(branch_count, delay_step, state=state)
    latency = engine_hard.latency_bits
    
    deint_hard = engine_hard.process_array(hard_bits).astype(np.uint8)
    
    deint_soft = None
    if soft_llrs is not None:
        # Clone current state before hard bit processing or duplicate engine
        # To maintain identical state progression for soft LLRs:
        engine_soft = ConvolutionalDeinterleaverEngine(branch_count, delay_step)
        deint_soft = engine_soft.process_array(soft_llrs).astype(np.float32)
        
    if trim_latency and len(deint_hard) > latency:
        deint_hard = deint_hard[latency:]
        if deint_soft is not None:
            deint_soft = deint_soft[latency:]
            
    return deint_hard, deint_soft, engine_hard.state, latency
