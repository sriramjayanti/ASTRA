"""
Tiny Overfit Test for ASTRA Modulation Models V2.
Verifies that ResNet-1D V2 and Spectrogram CNN-2D V2 can both overfit a tiny batch
before embarking on full training.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from astra_modulation_v2.class_schema import NUM_CLASSES_V2, MODULATION_CLASSES_V2
from astra_modulation_v2.models.resnet1d import ResNet1DV2
from astra_modulation_v2.models.spectrogram_cnn import SpectrogramCNN2DV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Tiny Overfit Test on device: {device}")

# 1. Create synthetic tiny batch
B = 16
N = 2048
labels = torch.randint(0, NUM_CLASSES_V2, (B,), device=device)
x_1d = torch.randn(B, 2, N, device=device)
x_2d = torch.randn(B, 1, 128, 128, device=device)

criterion = nn.CrossEntropyLoss()

# -------------------------------------------------------------
# Test 1: ResNet-1D V2 Overfit
# -------------------------------------------------------------
print("\n--- Testing ResNet-1D V2 Tiny Overfit ---")
model_1d = ResNet1DV2(in_channels=2, num_classes=NUM_CLASSES_V2).to(device)
optimizer_1d = optim.AdamW(model_1d.parameters(), lr=3e-3)

model_1d.train()
loss_1d = 0.0
acc_1d = 0.0
for step in range(35):
    optimizer_1d.zero_grad()
    logits, _ = model_1d(x_1d)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer_1d.step()
    
    preds = torch.argmax(logits, dim=1)
    acc = (preds == labels).float().mean().item()
    loss_1d = loss.item()
    acc_1d = acc
    if step % 10 == 0 or step == 34:
        print(f"Step {step:02d}: Loss = {loss_1d:.4f}, Accuracy = {acc_1d * 100:.1f}%")

assert acc_1d >= 0.90, f"ResNet-1D failed tiny overfit test! Acc: {acc_1d}"
print("[PASS] ResNet-1D V2 successfully overfit tiny batch!")

# -------------------------------------------------------------
# Test 2: Spectrogram CNN-2D V2 Overfit
# -------------------------------------------------------------
print("\n--- Testing Spectrogram CNN-2D V2 Tiny Overfit ---")
model_2d = SpectrogramCNN2DV2(in_channels=1, num_classes=NUM_CLASSES_V2).to(device)
optimizer_2d = optim.AdamW(model_2d.parameters(), lr=3e-3)

model_2d.train()
loss_2d = 0.0
acc_2d = 0.0
for step in range(35):
    optimizer_2d.zero_grad()
    logits, _ = model_2d(x_2d)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer_2d.step()
    
    preds = torch.argmax(logits, dim=1)
    acc = (preds == labels).float().mean().item()
    loss_2d = loss.item()
    acc_2d = acc
    if step % 10 == 0 or step == 34:
        print(f"Step {step:02d}: Loss = {loss_2d:.4f}, Accuracy = {acc_2d * 100:.1f}%")

assert acc_2d >= 0.90, f"Spectrogram CNN-2D failed tiny overfit test! Acc: {acc_2d}"
print("[PASS] Spectrogram CNN-2D V2 successfully overfit tiny batch!")
print("\nALL TINY OVERFIT TESTS PASSED! Proceeding to full training.")
