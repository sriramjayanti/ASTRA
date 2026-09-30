import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with two convolutional layers, batch normalization,
    and a skip (shortcut) connection for stable deep feature extraction.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=5, padding=2, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.skip = (
            nn.Identity()
            if (stride == 1 and in_channels == out_channels)
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1DModClassifier(nn.Module):
    """
    ASTRA 1D ResNet Modulation Classifier.
    
    Accepts raw IQ tensor [Batch, 2, N] where:
      channel 0: In-phase (I)
      channel 1: Quadrature (Q)
      
    Outputs raw logits [Batch, num_classes].
    """
    def __init__(
        self,
        num_classes: int = 8,
        input_channels: int = 2,
        base_channels: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.base_channels = base_channels
        
        # Initial stem: receptive field expansion & initial feature projection
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 4 Residual stages
        self.features = nn.Sequential(
            ResidualBlock1D(base_channels, base_channels, stride=1),
            ResidualBlock1D(base_channels, base_channels * 2, stride=2),
            ResidualBlock1D(base_channels * 2, base_channels * 4, stride=2),
            ResidualBlock1D(base_channels * 4, base_channels * 4, stride=2)
        )
        
        # Global adaptive average pooling makes it agnostic to input window length N
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: [B, 2, N]
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)  # [B, channels]
        x = self.dropout(x)
        logits = self.fc(x)           # [B, num_classes]
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: Optional[Dict[str, Any]] = None) -> ResNet1DModClassifier:
    """Instantiates the ResNet1D model from configuration dictionary or defaults."""
    if config is None:
        return ResNet1DModClassifier()
    
    m_cfg = config.get("model", {})
    return ResNet1DModClassifier(
        num_classes=m_cfg.get("num_classes", 8),
        input_channels=m_cfg.get("input_channels", 2),
        base_channels=m_cfg.get("base_channels", 64),
        dropout=m_cfg.get("dropout", 0.2)
    )


if __name__ == "__main__":
    model = create_model()
    x = torch.randn(4, 2, 2048)
    out = model(x)
    print(f"Model instantiated successfully.")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Trainable parameters: {model.count_parameters():,}")
