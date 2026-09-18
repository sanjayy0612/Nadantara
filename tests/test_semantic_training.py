from pathlib import Path

import torch

from seamtranslation.models import ModelConfig, SpeechSemanticModel
from seamtranslation.evaluation import semantic_generation_metrics
from seamtranslation.tokenization import HindiTokenizer
from seamtranslation.training import SemanticExample, collate_semantic_batch, semantic_loss
from train.train_tokenizer import train_tokenizer
from train.train_semantic import deterministic_batch_indices


def test_semantic_loss_backpropagates(tmp_path: Path) -> None:
    manifest = tmp_path / "tokens.jsonl"
    manifest.write_text(
        '{"target_text":"यह एक प्रशिक्षण वाक्य है"}\n'
        '{"target_text":"यह दूसरा वाक्य है"}\n'
    )
    tokenizer_dir = tmp_path / "tokenizer"
    train_tokenizer([manifest], tokenizer_dir, vocabulary_size=64, text_field="target_text")
    tokenizer = HindiTokenizer(tokenizer_dir / "hindi.model")
    config = ModelConfig(
        feature_bins=20,
        model_dim=32,
        encoder_layers=1,
        encoder_heads=4,
        encoder_ffn_dim=64,
        conformer_kernel_size=7,
        semantic_layers=1,
        semantic_heads=4,
        semantic_ffn_dim=64,
        semantic_vocab_size=tokenizer.vocabulary_size,
        unit_layers=1,
        unit_heads=4,
        unit_ffn_dim=64,
        unit_dim=32,
        unit_vocab_size=16,
        vocoder_channels=32,
        max_semantic_tokens=16,
        dropout=0.0,
        pad_token_id=tokenizer.special_ids.pad,
        bos_token_id=tokenizer.special_ids.bos,
        eos_token_id=tokenizer.special_ids.eos,
    )
    model = SpeechSemanticModel(config)
    batch = collate_semantic_batch(
        [SemanticExample(torch.zeros(1_600), "यह एक प्रशिक्षण वाक्य है")],
        tokenizer,
        max_tokens=config.max_semantic_tokens,
    )

    loss = semantic_loss(model, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.semantic_decoder.output_projection.weight.grad is not None


def test_deterministic_batches_resume_at_same_sequence() -> None:
    uninterrupted = [
        deterministic_batch_indices(step, dataset_size=5, batch_size=2, seed=17)
        for step in range(6)
    ]
    resumed = [
        deterministic_batch_indices(step, dataset_size=5, batch_size=2, seed=17)
        for step in range(3, 6)
    ]

    assert uninterrupted[3:] == resumed
    assert uninterrupted[0] != uninterrupted[3]


def test_generation_metrics_detect_repetition_and_missing_eos() -> None:
    metrics = semantic_generation_metrics(
        ["यह ठीक है", "और और और"],
        ["यह ठीक है", "यह अलग है"],
        [[4, 5, 6], [7, 7, 7]],
        [True, False],
    )

    assert metrics["examples"] == 2
    assert metrics["chrf2"] > 0
    assert metrics["bleu"] > 0
    assert metrics["eos_rate"] == 0.5
    assert metrics["repeated_token_rate"] == 2 / 6
