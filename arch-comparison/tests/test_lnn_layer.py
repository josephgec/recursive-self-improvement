"""Tests for LNN attention layer: forward pass, gating, gradients."""

from __future__ import annotations

import pytest
import torch

from src.integrative.lnn_layer import LNNAttentionLayer
from src.integrative.training import IntegrativeTrainer, TrainingResult


class TestLNNAttentionLayer:
    def test_forward_pass_shape(self):
        layer = LNNAttentionLayer(hidden_dim=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert out.shape == (2, 8, 64)

    def test_forward_single_head(self):
        layer = LNNAttentionLayer(hidden_dim=16, num_heads=1)
        x = torch.randn(1, 4, 16)
        out = layer(x)
        assert out.shape == (1, 4, 16)

    def test_forward_batch_size_one(self):
        layer = LNNAttentionLayer(hidden_dim=32, num_heads=4)
        x = torch.randn(1, 6, 32)
        out = layer(x)
        assert out.shape == (1, 6, 32)

    def test_forward_with_attention_mask(self):
        layer = LNNAttentionLayer(hidden_dim=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        mask = torch.ones(2, 8)
        mask[:, -2:] = 0  # Mask last 2 positions
        out = layer(x, attention_mask=mask)
        assert out.shape == (2, 8, 64)

    def test_gradients_flow(self):
        layer = LNNAttentionLayer(hidden_dim=32, num_heads=4)
        x = torch.randn(2, 4, 32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 4, 32)

    def test_parameter_gradients(self):
        layer = LNNAttentionLayer(hidden_dim=32, num_heads=4)
        x = torch.randn(2, 4, 32)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_classify_relations_shape(self):
        layer = LNNAttentionLayer(hidden_dim=32, num_heads=4)
        x = torch.randn(2, 4, 32)
        probs = layer._classify_relations(x)
        assert probs.shape == (2, 4, 4, 3)
        # Should be valid probabilities
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 4, 4), atol=1e-5)

    def test_compute_logical_bias_shape(self):
        layer = LNNAttentionLayer(hidden_dim=32, num_heads=4)
        probs = torch.randn(2, 4, 4, 3).softmax(dim=-1)
        bias = layer._compute_logical_bias(probs)
        assert bias.shape == (2, 4, 4, 4)  # (batch, heads, seq, seq)

    def test_gating_output_bounded(self):
        layer = LNNAttentionLayer(hidden_dim=32, num_heads=4)
        x = torch.randn(2, 4, 32)
        gate_vals = torch.sigmoid(layer.gate(x))
        assert (gate_vals >= 0).all()
        assert (gate_vals <= 1).all()

    def test_logical_bias_initial_zero(self):
        layer = LNNAttentionLayer(hidden_dim=32, num_heads=4)
        assert torch.allclose(layer.logical_bias, torch.zeros(4, 3))

    def test_different_hidden_dims(self):
        for hdim in [16, 32, 64, 128]:
            layer = LNNAttentionLayer(hidden_dim=hdim, num_heads=4)
            x = torch.randn(1, 4, hdim)
            out = layer(x)
            assert out.shape == (1, 4, hdim)

    def test_assertion_hidden_dim_divisibility(self):
        with pytest.raises(AssertionError):
            LNNAttentionLayer(hidden_dim=33, num_heads=4)

    def test_deterministic_output(self):
        layer = LNNAttentionLayer(hidden_dim=32, num_heads=4)
        layer.eval()
        x = torch.randn(1, 4, 32)
        out1 = layer(x)
        out2 = layer(x)
        assert torch.allclose(out1, out2)


# ── Training tests ──

class TestIntegrativeTrainer:
    def test_prepare_model(self):
        trainer = IntegrativeTrainer(hidden_dim=32, num_heads=4)
        model = trainer.prepare_model()
        assert model is not None
        assert isinstance(model, LNNAttentionLayer)

    def test_train_returns_result(self):
        trainer = IntegrativeTrainer(hidden_dim=16, num_heads=2, epochs=3)
        result = trainer.train()
        assert isinstance(result, TrainingResult)
        assert result.epochs_completed == 3
        assert len(result.loss_history) == 3

    def test_train_loss_decreases_or_bounded(self):
        trainer = IntegrativeTrainer(hidden_dim=16, num_heads=2, epochs=5, learning_rate=0.01)
        result = trainer.train()
        # Loss should be finite
        for loss in result.loss_history:
            assert loss >= 0
            assert not float("inf") == loss

    def test_evaluate_returns_accuracy(self):
        trainer = IntegrativeTrainer(hidden_dim=16, num_heads=2, epochs=2)
        trainer.train()
        accuracy = trainer.evaluate()
        assert 0.0 <= accuracy <= 1.0

    def test_evaluate_without_training(self):
        trainer = IntegrativeTrainer()
        accuracy = trainer.evaluate()
        assert accuracy == 0.0

    def test_train_auto_prepares_model(self):
        trainer = IntegrativeTrainer(hidden_dim=16, num_heads=2, epochs=1)
        assert trainer.model is None
        result = trainer.train()
        assert trainer.model is not None
        assert result.epochs_completed == 1

    def test_final_loss_is_last_in_history(self):
        trainer = IntegrativeTrainer(hidden_dim=16, num_heads=2, epochs=3)
        result = trainer.train()
        assert result.final_loss == result.loss_history[-1]
