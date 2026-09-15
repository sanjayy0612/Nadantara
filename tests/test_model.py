import io

import pytest

torch = pytest.importorskip("torch")
models = pytest.importorskip("seamtranslation.models")
ModelConfig = models.ModelConfig
SeamTranslationModel = models.SeamTranslationModel


def tiny_config() -> ModelConfig:
    return ModelConfig(
        feature_bins=20,
        model_dim=32,
        encoder_layers=1,
        encoder_heads=4,
        encoder_ffn_dim=64,
        conformer_kernel_size=7,
        semantic_layers=1,
        semantic_heads=4,
        semantic_ffn_dim=64,
        semantic_vocab_size=32,
        unit_layers=1,
        unit_heads=4,
        unit_ffn_dim=64,
        unit_dim=32,
        unit_vocab_size=16,
        vocoder_channels=32,
        max_semantic_tokens=8,
        dropout=0.0,
    )


def test_complete_training_graph_shapes() -> None:
    model = SeamTranslationModel(tiny_config()).eval()
    waveform = torch.randn(2, 1_600)
    lengths = torch.tensor([1_600, 1_280])
    semantic_tokens = torch.tensor([[1, 8, 9, 2], [1, 7, 2, 0]])
    durations = torch.tensor([[1, 2, 1, 1], [2, 1, 1, 0]])

    output = model(waveform, lengths, semantic_tokens, unit_durations=durations)

    assert output.semantic_logits.shape == (2, 4, 32)
    assert output.unit_logits.shape == (2, 5, 16)
    assert output.predicted_durations.shape == (2, 4)
    assert output.waveform.shape == (2, 1_600)


def test_state_reset_and_serialization_are_deterministic() -> None:
    config = tiny_config()
    model = SeamTranslationModel(config).eval()
    waveform = torch.randn(1, 1_600)
    lengths = torch.tensor([1_600])
    tokens = torch.tensor([[1, 4, 2]])
    durations = torch.ones_like(tokens)

    expected = model(waveform, lengths, tokens, unit_durations=durations).semantic_logits
    model.reset_state()
    repeated = model(waveform, lengths, tokens, unit_durations=durations).semantic_logits
    assert torch.equal(expected, repeated)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = SeamTranslationModel(config).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    actual = restored(waveform, lengths, tokens, unit_durations=durations).semantic_logits
    assert torch.equal(expected, actual)


def test_invalid_head_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="model_dim must be divisible"):
        ModelConfig(model_dim=30, encoder_heads=4)
