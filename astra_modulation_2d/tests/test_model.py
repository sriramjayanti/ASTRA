import torch

from astra_modulation_2d.src.model import ASTRASpectrogramCNN


def test_7_model_input_shape_accepted():
    """TEST 7: Model accepts both 4D spectrograms [B, 1, F, T] and 3D raw IQ [B, 2, N]."""
    classes = ["bpsk", "qpsk", "8psk", "16qam"]
    model = ASTRASpectrogramCNN(class_names=classes)

    # 1. Direct Spectrogram Input [2, 1, 128, 65]
    spec_input = torch.randn(2, 1, 128, 65)
    out_spec = model(spec_input)
    assert out_spec.shape == (2, len(classes))

    # 2. Raw IQ Input [2, 2, 2048]
    iq_input = torch.randn(2, 2, 2048)
    out_iq = model(iq_input)
    assert out_iq.shape == (2, len(classes))


def test_8_model_output_logits():
    """TEST 8: Model outputs unnormalized class logits of shape [B, num_classes]."""
    classes = ["2fsk", "4fsk", "bpsk", "qpsk", "8psk", "dqpsk", "msk", "16qam", "64qam", "256qam"]
    model = ASTRASpectrogramCNN(class_names=classes)
    x = torch.randn(4, 2, 2048)
    logits = model(x)
    assert logits.shape == (4, 10)
    assert logits.dtype == torch.float32


def test_9_extract_features_shape():
    """TEST 9: extract_features produces [B, 256] embedding vector."""
    classes = ["bpsk", "qpsk"]
    model = ASTRASpectrogramCNN(class_names=classes, embedding_dim=256)
    x = torch.randn(5, 2, 2048)
    feat = model.extract_features(x)
    assert feat.shape == (5, 256), f"Extracted feature shape was {feat.shape}, expected (5, 256)"


def test_20_variable_batch_size():
    """TEST 20: Model seamlessly handles variable batch sizes from 1 to 64."""
    classes = ["bpsk", "qpsk", "8psk"]
    model = ASTRASpectrogramCNN(class_names=classes)
    model.eval()
    with torch.no_grad():
        for b_size in [1, 2, 7, 16, 32]:
            x = torch.randn(b_size, 2, 2048)
            out = model(x)
            assert out.shape == (b_size, len(classes))
