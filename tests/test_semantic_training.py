from pathlib import Path

import torch

from seamtranslation.models import ModelConfig, SeamTranslationModel
from seamtranslation.tokenization import HindiTokenizer
from seamtranslation.training import SemanticExample, collate_semantic_batch, semantic_loss
from train.train_tokenizer import train_tokenizer


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
    model = SeamTranslationModel(config)
    batch = collate_semantic_batch(
        [SemanticExample(torch.zeros(1_600), "यह एक प्रशिक्षण वाक्य है")],
        tokenizer,
        max_tokens=config.max_semantic_tokens,
    )

    loss = semantic_loss(model, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.semantic_decoder.output_projection.weight.grad is not None
