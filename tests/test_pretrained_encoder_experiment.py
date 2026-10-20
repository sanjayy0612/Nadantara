import torch

from seamtranslation.models import ModelConfig
from train.compare_pretrained_encoder import CachedSpeechHead, memory_batch


def test_cached_memory_masks_padding_and_preserves_valid_features():
    features = [torch.ones(2, 8), torch.full((4, 8), 2.0)]
    memory, padding = memory_batch(features, [0, 1], "cpu")
    assert memory.shape == (2, 4, 8)
    assert padding.tolist() == [[False, False, True, True], [False] * 4]
    assert torch.equal(memory[0, :2], features[0])


def test_head_ignores_padded_memory_and_trains_adapter():
    config = ModelConfig(model_dim=16, encoder_heads=4, semantic_heads=4,
                         semantic_layers=1, semantic_ffn_dim=32,
                         semantic_vocab_size=20, max_semantic_tokens=8, dropout=0.0)
    head = CachedSpeechHead(config, 8).eval()
    features, padding = memory_batch([torch.randn(2, 8), torch.randn(4, 8)], [0, 1], "cpu")
    tokens = torch.tensor([[1, 4], [1, 5]])
    before = head(features, padding, tokens)
    changed = features.clone()
    changed[0, 2:] = 1000
    after = head(changed, padding, tokens)
    assert torch.allclose(before, after, atol=1e-6)
    before.sum().backward()
    assert head.adapter[0].weight.grad is not None
    assert torch.isfinite(head.adapter[0].weight.grad).all()
