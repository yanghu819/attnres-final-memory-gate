import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('AUTORESEARCH_ATTN_BACKEND', 'sdpa')
os.environ.setdefault('AUTORESEARCH_USE_COMPILE', '0')

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

import train
from autoresearch_attnres_project.metrics import parse_required_summary_metrics


class AttnResProjectedTests(unittest.TestCase):
    def test_depth_mixer_source_key_scale_changes_weights(self):
        mixer = train.DepthSoftmaxMixer(num_queries=1, dim=4)
        mixer.init_weights()
        mixer.queries.data[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        sources = torch.tensor(
            [
                [[[1.0, 0.0, 0.0, 0.0]]],
                [[[0.0, 1.0, 0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        _, weights_base, _ = mixer.mix_with_weights(0, sources, include_registers=False)
        _, weights_scaled, _ = mixer.mix_with_weights(
            0,
            sources,
            include_registers=False,
            source_key_scale=torch.tensor([[[0.0, 1.0, 1.0, 1.0]]]),
        )
        self.assertGreater(float(weights_base[0, 0, 0]), 0.5)
        self.assertAlmostEqual(float(weights_scaled[0, 0, 0]), 0.5, places=6)

    def test_depth_mixer_source_value_scale_changes_mixed_output(self):
        mixer = train.DepthSoftmaxMixer(num_queries=1, dim=4)
        mixer.init_weights()
        sources = torch.tensor(
            [
                [[[1.0, 0.0, 0.0, 0.0]]],
                [[[0.0, 1.0, 0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        mixed_base, _, _ = mixer.mix_with_weights(0, sources, include_registers=False)
        mixed_scaled, _, _ = mixer.mix_with_weights(
            0,
            sources,
            include_registers=False,
            source_value_scale=torch.tensor([[[1.0, 2.0, 1.0, 1.0]]]),
        )
        self.assertTrue(torch.allclose(mixed_base, torch.tensor([[[0.5, 0.5, 0.0, 0.0]]]), atol=1e-6, rtol=0.0))
        self.assertTrue(torch.allclose(mixed_scaled, torch.tensor([[[0.5, 1.0, 0.0, 0.0]]]), atol=1e-6, rtol=0.0))

    def test_depth_mixer_source_value_residual_changes_mixed_output(self):
        mixer = train.DepthSoftmaxMixer(num_queries=1, dim=4)
        mixer.init_weights()
        sources = torch.tensor(
            [
                [[[1.0, 0.0, 0.0, 0.0]]],
                [[[0.0, 1.0, 0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        mixed_base, _, _ = mixer.mix_with_weights(0, sources, include_registers=False)
        mixed_resid, _, _ = mixer.mix_with_weights(
            0,
            sources,
            include_registers=False,
            source_value_residual=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        )
        self.assertTrue(torch.allclose(mixed_base, torch.tensor([[[0.5, 0.5, 0.0, 0.0]]]), atol=1e-6, rtol=0.0))
        # value residuals are applied on top of RMS-normalized source values.
        # For a one-hot vector in dim=4, rms_norm scales the active entry to 2.0.
        # With uniform 0.5 / 0.5 routing, the residual contributes +1.0 on dim 0.
        self.assertTrue(torch.allclose(mixed_resid, torch.tensor([[[1.5, 0.5, 0.0, 0.0]]]), atol=1e-6, rtol=0.0))

    def test_depth_mixer_source_logit_bias_changes_weights(self):
        mixer = train.DepthSoftmaxMixer(num_queries=1, dim=4)
        mixer.init_weights()
        sources = torch.tensor(
            [
                [[[1.0, 0.0, 0.0, 0.0]]],
                [[[0.0, 1.0, 0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        _, weights_base, _ = mixer.mix_with_weights(0, sources, include_registers=False)
        _, weights_biased, _ = mixer.mix_with_weights(
            0,
            sources,
            include_registers=False,
            source_logit_bias=torch.tensor([0.0, 2.0]),
        )
        self.assertAlmostEqual(float(weights_base[0, 0, 0]), 0.5, places=6)
        self.assertGreater(float(weights_biased[1, 0, 0]), float(weights_base[1, 0, 0]))

    def test_final_memory_static_gate_matches_cap_times_sigmoid_bias(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            window_pattern='L',
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='static',
            attnres_final_memory_value_mode='raw',
            attnres_final_memory_scale=1.0,
            attnres_final_memory_gate_cap=0.85,
            attnres_final_memory_gate_bias_init=4.0,
        )
        model = train.GPT(cfg)
        model.init_weights()
        idx = torch.tensor([[1, 2]], dtype=torch.long)
        x = torch.zeros(1, 2, 8)
        _, stats = model._apply_attnres_final_memory(idx, x)
        expected = 0.85 * torch.sigmoid(torch.tensor(4.0))
        self.assertAlmostEqual(float(stats["final_memory_gate"]), float(expected), places=6)

    def test_zero_dynamic_final_memory_reduces_to_static_gate(self):
        common = dict(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            window_pattern='L',
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_value_mode='raw',
            attnres_final_memory_scale=1.0,
            attnres_final_memory_gate_cap=0.85,
            attnres_final_memory_gate_bias_init=4.0,
        )
        static_model = train.GPT(train.GPTConfig(**common, attnres_final_memory_mode='static'))
        dynamic_model = train.GPT(train.GPTConfig(**common, attnres_final_memory_mode='dynamic'))
        static_model.init_weights()
        dynamic_model.init_weights()
        dynamic_model.load_state_dict(static_model.state_dict(), strict=False)
        dynamic_model.attnres_final_memory_gate_proj.weight.data.zero_()

        idx = torch.tensor([[1, 2]], dtype=torch.long)
        x = torch.randn(1, 2, 8)
        out_static, stats_static = static_model._apply_attnres_final_memory(idx, x)
        out_dynamic, stats_dynamic = dynamic_model._apply_attnres_final_memory(idx, x)
        self.assertTrue(torch.allclose(out_static, out_dynamic, atol=1e-6, rtol=0.0))
        self.assertAlmostEqual(float(stats_static["final_memory_gate"]), float(stats_dynamic["final_memory_gate"]), places=6)

    def test_hybrid_static_uses_separate_query_and_blend_gates(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            window_pattern='L',
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='hybrid_static',
            attnres_final_memory_source='full',
            attnres_final_memory_value_mode='raw',
            attnres_final_memory_scale=1.0,
            attnres_final_memory_gate_cap=0.85,
            attnres_final_memory_gate_bias_init=4.0,
            attnres_final_memory_query_cap=0.10,
            attnres_final_memory_query_bias_init=0.0,
        )
        model = train.GPT(cfg)
        model.init_weights()
        idx = torch.tensor([[1, 2]], dtype=torch.long)
        x = torch.zeros(1, 2, 8)
        _, blend_stats = model._apply_attnres_final_memory(idx, x)
        _, query_stats = model._build_attnres_final_memory_query(idx, x)
        self.assertAlmostEqual(float(blend_stats["final_memory_gate"]), float(0.85 * torch.sigmoid(torch.tensor(4.0))), places=4)
        self.assertAlmostEqual(float(query_stats["final_memory_query_gate"]), float(0.10 * torch.sigmoid(torch.tensor(0.0))), places=4)

    def test_zero_hybrid_projected_query_reduces_to_hybrid_static_query(self):
        common = dict(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            window_pattern='L',
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_source='full',
            attnres_final_memory_value_mode='raw',
            attnres_final_memory_scale=1.0,
            attnres_final_memory_gate_cap=0.85,
            attnres_final_memory_gate_bias_init=4.0,
            attnres_final_memory_query_cap=0.10,
            attnres_final_memory_query_bias_init=0.0,
            attnres_final_memory_query_delta_scale=1.0,
        )
        static_model = train.GPT(train.GPTConfig(**common, attnres_final_memory_mode='hybrid_static'))
        projected_model = train.GPT(train.GPTConfig(**common, attnres_final_memory_mode='hybrid_projected'))
        static_model.init_weights()
        projected_model.init_weights()
        projected_model.load_state_dict(static_model.state_dict(), strict=False)
        projected_model.attnres_final_memory_query_proj.weight.data.zero_()
        idx = torch.tensor([[1, 2]], dtype=torch.long)
        x = torch.randn(1, 2, 8)
        query_static, stats_static = static_model._build_attnres_final_memory_query(idx, x)
        query_proj, stats_proj = projected_model._build_attnres_final_memory_query(idx, x)
        self.assertTrue(torch.allclose(query_static, query_proj, atol=1e-6, rtol=0.0))
        self.assertAlmostEqual(float(stats_static["final_memory_query_gate"]), float(stats_proj["final_memory_query_gate"]), places=6)

    def test_zero_modkv_reduces_to_identity_modulation(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            window_pattern='L',
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='modkv',
            attnres_final_memory_source='tied_bigram',
            attnres_final_memory_bigram_buckets=16,
            attnres_final_memory_bigram_banks=2,
            attnres_final_memory_value_mode='raw',
            attnres_final_memory_scale=1.0,
            attnres_final_memory_k_mod_scale=0.05,
            attnres_final_memory_v_mod_scale=0.05,
        )
        model = train.GPT(cfg)
        model.init_weights()
        idx = torch.tensor([[1, 2]], dtype=torch.long)
        q_delta, logit_bias, key_scale, value_scale, value_residual, stats = model._build_attnres_final_memory_modulation(idx)
        self.assertIsNone(q_delta)
        self.assertIsNone(logit_bias)
        self.assertIsNone(value_residual)
        self.assertTrue(torch.allclose(key_scale, torch.ones_like(key_scale), atol=1e-6, rtol=0.0))
        self.assertTrue(torch.allclose(value_scale, torch.ones_like(value_scale), atol=1e-6, rtol=0.0))
        self.assertAlmostEqual(float(stats["final_memory_mod_k_delta_norm"]), 0.0, places=6)
        self.assertAlmostEqual(float(stats["final_memory_mod_v_delta_norm"]), 0.0, places=6)

    def test_zero_vres_reduces_to_identity_residual(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            window_pattern='L',
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='vres',
            attnres_final_memory_source='tied',
            attnres_final_memory_value_mode='raw',
            attnres_final_memory_scale=1.0,
            attnres_final_memory_vres_scale=0.05,
        )
        model = train.GPT(cfg)
        model.init_weights()
        idx = torch.tensor([[1, 2]], dtype=torch.long)
        q_delta, logit_bias, key_scale, value_scale, value_residual, stats = model._build_attnres_final_memory_modulation(idx)
        self.assertIsNone(q_delta)
        self.assertIsNone(logit_bias)
        self.assertIsNone(key_scale)
        self.assertIsNone(value_scale)
        self.assertTrue(torch.allclose(value_residual, torch.zeros_like(value_residual), atol=1e-6, rtol=0.0))
        self.assertAlmostEqual(float(stats["final_memory_vres_delta_norm"]), 0.0, places=6)

    def test_final_memory_mode_requires_no_registers(self):
        with self.assertRaises(ValueError):
            train.GPT(
                train.GPTConfig(
                    sequence_len=8,
                    vocab_size=16,
                    n_layer=2,
                    n_head=2,
                    n_kv_head=2,
                    n_embd=8,
                    dropout=0.0,
                    bias=False,
                    use_value_embeds=False,
                    use_rotary=False,
                    use_qk_norm=False,
                    use_absolute_positions=True,
                    norm_type='layernorm',
                    mlp_activation='gelu',
                    reference_init=True,
                    reference_optimizer=True,
                    attnres_mode='block',
                    attnres_block_size=2,
                    attnres_weight_mode='softmax',
                    attnres_num_registers=0,
                    attnres_token_registers=1,
                    attnres_token_register_mode='shared',
                    attnres_final_memory_mode='dynamic',
                )
            )

    def test_grouped_final_memory_applies_per_group_gates(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            window_pattern='L',
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='static',
            attnres_final_memory_value_mode='raw',
            attnres_final_memory_scale=1.0,
            attnres_final_memory_groups=2,
            attnres_final_memory_gate_cap=0.8,
            attnres_final_memory_gate_bias_init=0.0,
        )
        model = train.GPT(cfg)
        model.init_weights()
        model.attnres_final_memory_gate_bias.data.copy_(torch.tensor([0.0, 4.0]))
        model.attnres_final_memory_embed.weight.data.zero_()
        model.attnres_final_memory_embed.weight.data[1, :4] = 1.0
        model.attnres_final_memory_embed.weight.data[1, 4:] = 2.0

        idx = torch.tensor([[1]], dtype=torch.long)
        x = torch.zeros(1, 1, 8)
        out, stats = model._apply_attnres_final_memory(idx, x)
        g0 = 0.8 * torch.sigmoid(torch.tensor(0.0))
        g1 = 0.8 * torch.sigmoid(torch.tensor(4.0))
        expected = torch.tensor([[[g0, g0, g0, g0, 2 * g1, 2 * g1, 2 * g1, 2 * g1]]], dtype=out.dtype)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6, rtol=0.0))
        self.assertAlmostEqual(float(stats["final_memory_gate"]), float(torch.tensor([g0, g1]).mean()), places=6)

    def test_factorized_final_memory_uses_rank_projection(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            window_pattern='L',
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='static',
            attnres_final_memory_source='factorized',
            attnres_final_memory_rank=2,
            attnres_final_memory_value_mode='raw',
            attnres_final_memory_scale=1.0,
            attnres_final_memory_gate_cap=1.0,
            attnres_final_memory_gate_bias_init=10.0,
        )
        model = train.GPT(cfg)
        model.init_weights()
        model.attnres_final_memory_embed.weight.data.zero_()
        model.attnres_final_memory_embed.weight.data[1] = torch.tensor([3.0, 5.0])
        model.attnres_final_memory_proj.weight.data.zero_()
        model.attnres_final_memory_proj.weight.data[:, 0] = 1.0
        model.attnres_final_memory_proj.weight.data[:, 1] = 2.0

        idx = torch.tensor([[1]], dtype=torch.long)
        memory = model._build_attnres_final_memory(idx)
        expected = torch.full((1, 1, 8), 13.0, dtype=memory.dtype)
        self.assertTrue(torch.allclose(memory, expected, atol=1e-6, rtol=0.0))

    def test_tied_final_memory_reuses_input_embedding_table(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            window_pattern='L',
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='static',
            attnres_final_memory_source='tied',
            attnres_final_memory_value_mode='raw',
            attnres_final_memory_scale=1.0,
            attnres_final_memory_gate_cap=1.0,
            attnres_final_memory_gate_bias_init=10.0,
        )
        model = train.GPT(cfg)
        model.init_weights()
        model.transformer.wte.weight.data.zero_()
        model.transformer.wte.weight.data[2] = torch.arange(8, dtype=model.transformer.wte.weight.dtype)
        idx = torch.tensor([[2]], dtype=torch.long)
        memory = model._build_attnres_final_memory(idx)
        expected = torch.arange(8, dtype=memory.dtype).view(1, 1, 8)
        self.assertTrue(torch.allclose(memory, expected, atol=1e-6, rtol=0.0))

    def test_tied_residual_zero_init_matches_tied_memory(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            window_pattern='L',
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='static',
            attnres_final_memory_source='tied_residual',
            attnres_final_memory_residual_rank=4,
            attnres_final_memory_residual_scale=1.0,
            attnres_final_memory_value_mode='raw',
            attnres_final_memory_scale=1.0,
            attnres_final_memory_gate_cap=1.0,
            attnres_final_memory_gate_bias_init=10.0,
        )
        model = train.GPT(cfg)
        model.init_weights()
        model.transformer.wte.weight.data.zero_()
        model.transformer.wte.weight.data[3] = torch.arange(8, dtype=model.transformer.wte.weight.dtype)
        idx = torch.tensor([[3]], dtype=torch.long)
        memory = model._build_attnres_final_memory(idx)
        expected = torch.arange(8, dtype=memory.dtype).view(1, 1, 8)
        self.assertTrue(torch.allclose(memory, expected, atol=1e-6, rtol=0.0))

    def test_tied_bigram_zero_init_matches_tied_memory(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='static',
            attnres_final_memory_source='tied_bigram',
            attnres_final_memory_bigram_buckets=16,
            attnres_final_memory_bigram_scale=1.0,
            attnres_final_memory_value_mode='raw',
            attnres_final_memory_scale=1.0,
            attnres_final_memory_gate_cap=1.0,
            attnres_final_memory_gate_bias_init=10.0,
        )
        model = train.GPT(cfg)
        model.init_weights()
        model.transformer.wte.weight.data.zero_()
        model.transformer.wte.weight.data[4] = torch.arange(8, dtype=model.transformer.wte.weight.dtype)
        idx = torch.tensor([[4]], dtype=torch.long)
        memory = model._build_attnres_final_memory(idx)
        expected = torch.arange(8, dtype=memory.dtype).view(1, 1, 8)
        self.assertTrue(torch.allclose(memory, expected, atol=1e-6, rtol=0.0))

    def test_tied_bigram_multibank_zero_init_matches_tied_memory(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='static',
            attnres_final_memory_source='tied_bigram',
            attnres_final_memory_bigram_buckets=16,
            attnres_final_memory_bigram_banks=3,
            attnres_final_memory_bigram_scale=1.0,
            attnres_final_memory_value_mode='raw',
            attnres_final_memory_scale=1.0,
            attnres_final_memory_gate_cap=1.0,
            attnres_final_memory_gate_bias_init=10.0,
        )
        model = train.GPT(cfg)
        model.init_weights()
        model.transformer.wte.weight.data.zero_()
        model.transformer.wte.weight.data[4] = torch.arange(8, dtype=model.transformer.wte.weight.dtype)
        idx = torch.tensor([[4]], dtype=torch.long)
        memory = model._build_attnres_final_memory(idx)
        expected = torch.arange(8, dtype=memory.dtype).view(1, 1, 8)
        self.assertTrue(torch.allclose(memory, expected, atol=1e-6, rtol=0.0))

    def test_tied_bigram_factorized_zero_init_matches_tied_memory(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='static',
            attnres_final_memory_source='tied_bigram_factorized',
            attnres_final_memory_rank=4,
            attnres_final_memory_bigram_buckets=16,
            attnres_final_memory_bigram_banks=2,
            attnres_final_memory_bigram_scale=1.0,
            attnres_final_memory_value_mode='raw',
            attnres_final_memory_scale=1.0,
            attnres_final_memory_gate_cap=1.0,
            attnres_final_memory_gate_bias_init=10.0,
        )
        model = train.GPT(cfg)
        model.init_weights()
        model.transformer.wte.weight.data.zero_()
        model.transformer.wte.weight.data[4] = torch.arange(8, dtype=model.transformer.wte.weight.dtype)
        idx = torch.tensor([[4]], dtype=torch.long)
        memory = model._build_attnres_final_memory(idx)
        expected = torch.arange(8, dtype=memory.dtype).view(1, 1, 8)
        self.assertTrue(torch.allclose(memory, expected, atol=1e-6, rtol=0.0))

    def test_tied_bigram_factorized_adds_projected_residual(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='static',
            attnres_final_memory_source='tied_bigram_factorized',
            attnres_final_memory_rank=2,
            attnres_final_memory_bigram_buckets=16,
            attnres_final_memory_bigram_banks=2,
            attnres_final_memory_bigram_scale=1.0,
            attnres_final_memory_value_mode='raw',
            attnres_final_memory_scale=1.0,
            attnres_final_memory_gate_cap=1.0,
            attnres_final_memory_gate_bias_init=10.0,
        )
        model = train.GPT(cfg)
        model.init_weights()
        model.transformer.wte.weight.data.zero_()
        model.transformer.wte.weight.data[5] = torch.arange(8, dtype=model.transformer.wte.weight.dtype)
        for emb in model.attnres_final_memory_embed:
            emb.weight.data.zero_()
        b0 = model._attnres_bigram_bucket_ids(torch.tensor([[5]], dtype=torch.long), 0).item()
        b1 = model._attnres_bigram_bucket_ids(torch.tensor([[5]], dtype=torch.long), 1).item()
        model.attnres_final_memory_embed[0].weight.data[b0] = torch.tensor([1.0, 2.0], dtype=model.attnres_final_memory_embed[0].weight.dtype)
        model.attnres_final_memory_embed[1].weight.data[b1] = torch.tensor([3.0, 4.0], dtype=model.attnres_final_memory_embed[1].weight.dtype)
        model.attnres_final_memory_proj.weight.data.zero_()
        model.attnres_final_memory_proj.weight.data[:, 0] = 1.0
        model.attnres_final_memory_proj.weight.data[:, 1] = 0.5
        idx = torch.tensor([[5]], dtype=torch.long)
        memory = model._build_attnres_final_memory(idx)
        residual_scalar = (1.0 + 3.0) + 0.5 * (2.0 + 4.0)
        expected = torch.arange(8, dtype=memory.dtype).view(1, 1, 8) + residual_scalar
        self.assertTrue(torch.allclose(memory, expected, atol=1e-6, rtol=0.0))

    def test_tied_bigram_only_zero_init_is_zero_memory(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='static',
            attnres_final_memory_source='tied_bigram_only',
            attnres_final_memory_bigram_buckets=16,
            attnres_final_memory_bigram_banks=2,
            attnres_final_memory_bigram_scale=1.0,
            attnres_final_memory_value_mode='raw',
            attnres_final_memory_scale=1.0,
            attnres_final_memory_gate_cap=1.0,
            attnres_final_memory_gate_bias_init=10.0,
        )
        model = train.GPT(cfg)
        model.init_weights()
        idx = torch.tensor([[4]], dtype=torch.long)
        memory = model._build_attnres_final_memory(idx)
        self.assertTrue(torch.allclose(memory, torch.zeros_like(memory), atol=1e-6, rtol=0.0))

    def test_tied_bigram_only_omits_unigram_base(self):
        common = dict(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='static',
            attnres_final_memory_bigram_buckets=16,
            attnres_final_memory_bigram_scale=1.0,
            attnres_final_memory_value_mode='raw',
            attnres_final_memory_scale=1.0,
            attnres_final_memory_gate_cap=1.0,
            attnres_final_memory_gate_bias_init=10.0,
        )
        tied_model = train.GPT(train.GPTConfig(**common, attnres_final_memory_source='tied_bigram'))
        bigram_only_model = train.GPT(train.GPTConfig(**common, attnres_final_memory_source='tied_bigram_only'))
        tied_model.init_weights()
        bigram_only_model.init_weights()
        tied_model.transformer.wte.weight.data.zero_()
        bigram_only_model.transformer.wte.weight.data.zero_()
        tied_model.transformer.wte.weight.data[4] = torch.arange(8, dtype=tied_model.transformer.wte.weight.dtype)
        bigram_only_model.transformer.wte.weight.data[4] = torch.arange(8, dtype=bigram_only_model.transformer.wte.weight.dtype)
        idx = torch.tensor([[4]], dtype=torch.long)
        tied_memory = tied_model._build_attnres_final_memory(idx)
        bigram_only_memory = bigram_only_model._build_attnres_final_memory(idx)
        expected = torch.arange(8, dtype=tied_memory.dtype).view(1, 1, 8)
        self.assertTrue(torch.allclose(tied_memory, expected, atol=1e-6, rtol=0.0))
        self.assertTrue(torch.allclose(bigram_only_memory, torch.zeros_like(bigram_only_memory), atol=1e-6, rtol=0.0))

    def test_input_bigram_zero_init_is_identity(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_input_memory_mode='bigram',
            attnres_input_memory_value_mode='raw',
            attnres_input_memory_hash_dim=8,
            attnres_input_memory_bigram_buckets=16,
            attnres_input_memory_bigram_banks=2,
        )
        model = train.GPT(cfg)
        model.init_weights()
        idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
        x = model.transformer.wte(idx)
        x_mem, stats = model._apply_attnres_input_memory(idx, x)
        self.assertTrue(torch.allclose(x_mem, x, atol=1e-6, rtol=0.0))
        self.assertAlmostEqual(float(stats["input_memory_delta_norm"]), 0.0, places=6)

    def test_input_bigram_hash_adds_projected_residual(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_input_memory_mode='bigram',
            attnres_input_memory_value_mode='raw',
            attnres_input_memory_hash_dim=8,
            attnres_input_memory_bigram_buckets=32,
            attnres_input_memory_bigram_banks=1,
        )
        model = train.GPT(cfg)
        model.init_weights()
        model.attnres_input_bigram_proj.weight.data.copy_(torch.eye(8, dtype=model.attnres_input_bigram_proj.weight.dtype))
        idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
        bucket = int(model._attnres_input_bigram_bucket_ids(idx, 0)[0, 1].item())
        vec = torch.arange(8, dtype=model.attnres_input_bigram_embed.weight.dtype)
        model.attnres_input_bigram_embed.weight.data.zero_()
        model.attnres_input_bigram_embed.weight.data[bucket] = vec
        x = model.transformer.wte(idx)
        x_mem, _ = model._apply_attnres_input_memory(idx, x)
        expected = x.clone()
        expected[0, 1] = expected[0, 1] + vec.to(dtype=expected.dtype)
        self.assertTrue(torch.allclose(x_mem, expected, atol=5e-2, rtol=0.0))

    def test_input_bigram_trigram_combines_both_sources(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_input_memory_mode='bigram_trigram',
            attnres_input_memory_value_mode='raw',
            attnres_input_memory_hash_dim=8,
            attnres_input_memory_bigram_buckets=1024,
            attnres_input_memory_bigram_banks=1,
            attnres_input_memory_trigram_buckets=1024,
            attnres_input_memory_trigram_banks=1,
        )
        model = train.GPT(cfg)
        model.init_weights()
        model.attnres_input_bigram_proj.weight.data.copy_(torch.eye(8, dtype=model.attnres_input_bigram_proj.weight.dtype))
        model.attnres_input_trigram_proj.weight.data.copy_(torch.eye(8, dtype=model.attnres_input_trigram_proj.weight.dtype))
        idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
        bigram_bucket = int(model._attnres_input_bigram_bucket_ids(idx, 0)[0, 2].item())
        trigram_bucket = int(model._attnres_input_trigram_bucket_ids(idx, 0)[0, 2].item())
        bigram_vec = torch.ones(8, dtype=model.attnres_input_bigram_embed.weight.dtype)
        trigram_vec = torch.full((8,), 2.0, dtype=model.attnres_input_trigram_embed.weight.dtype)
        model.attnres_input_bigram_embed.weight.data.zero_()
        model.attnres_input_trigram_embed.weight.data.zero_()
        model.attnres_input_bigram_embed.weight.data[bigram_bucket] = bigram_vec
        model.attnres_input_trigram_embed.weight.data[trigram_bucket] = trigram_vec
        x = model.transformer.wte(idx)
        x_mem, stats = model._apply_attnres_input_memory(idx, x)
        expected = x.clone()
        bigram_ids = model._attnres_input_bigram_bucket_ids(idx, 0)
        trigram_ids = model._attnres_input_trigram_bucket_ids(idx, 0)
        expected = expected + (bigram_ids == bigram_bucket).unsqueeze(-1).to(dtype=expected.dtype) * bigram_vec.to(dtype=expected.dtype)
        expected = expected + (trigram_ids == trigram_bucket).unsqueeze(-1).to(dtype=expected.dtype) * trigram_vec.to(dtype=expected.dtype)
        self.assertTrue(torch.allclose((x_mem - x).float(), (expected - x).float(), atol=5e-2, rtol=0.0))
        self.assertIn('input_memory_trigram_rms', stats)

    def test_input_smear_gate_only_smears_residual(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_input_memory_mode='bigram',
            attnres_input_memory_value_mode='raw',
            attnres_input_memory_hash_dim=8,
            attnres_input_memory_bigram_buckets=32,
            attnres_input_memory_bigram_banks=1,
            attnres_input_memory_smear=True,
            attnres_input_memory_smear_bias_init=10.0,
        )
        model = train.GPT(cfg)
        model.init_weights()
        model.attnres_input_bigram_proj.weight.data.copy_(torch.eye(8, dtype=model.attnres_input_bigram_proj.weight.dtype))
        idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
        bucket = int(model._attnres_input_bigram_bucket_ids(idx, 0)[0, 1].item())
        vec = torch.arange(8, dtype=model.attnres_input_bigram_embed.weight.dtype)
        model.attnres_input_bigram_embed.weight.data.zero_()
        model.attnres_input_bigram_embed.weight.data[bucket] = vec
        x = model.transformer.wte(idx)
        x_mem, stats = model._apply_attnres_input_memory(idx, x)
        expected = x.clone()
        expected[0, 2] = expected[0, 2] + vec.to(dtype=expected.dtype)
        self.assertTrue(torch.allclose(x_mem, expected, atol=1e-4, rtol=0.0))
        self.assertGreater(float(stats["input_memory_smear_gate"]), 0.99)

    def test_unified_static_final_memory_logs_memory_source_weights(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=16,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            window_pattern='L',
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='unified_static',
            attnres_final_memory_source='tied_bigram',
            attnres_final_memory_value_mode='rmsnorm',
            attnres_final_memory_scale=0.25,
            attnres_final_memory_groups=1,
            attnres_final_memory_bigram_buckets=64,
            attnres_final_memory_bigram_banks=2,
            attnres_final_memory_gate_cap=0.85,
            attnres_final_memory_gate_bias_init=4.0,
        )
        model = train.GPT(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))
        loss = model(idx, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('memory_total', model.last_attnres_metrics)
        self.assertIn('memory_unigram', model.last_attnres_metrics)
        self.assertIn('memory_bigram_bank0', model.last_attnres_metrics)
        self.assertIn('memory_bigram_bank1', model.last_attnres_metrics)

    def test_unified_projected_final_memory_forward(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=16,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            window_pattern='L',
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='unified_projected',
            attnres_final_memory_source='tied_bigram_only',
            attnres_final_memory_value_mode='rmsnorm',
            attnres_final_memory_scale=0.25,
            attnres_final_memory_groups=1,
            attnres_final_memory_bigram_buckets=64,
            attnres_final_memory_bigram_banks=2,
            attnres_final_memory_gate_cap=0.85,
            attnres_final_memory_gate_bias_init=4.0,
        )
        model = train.GPT(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))
        loss = model(idx, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.attnres_final_memory_unified_proj)
        self.assertIn('memory_total', model.last_attnres_metrics)

    def test_query_static_final_memory_logs_query_stats(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=16,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            window_pattern='L',
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='query_static',
            attnres_final_memory_source='tied_bigram',
            attnres_final_memory_value_mode='rmsnorm',
            attnres_final_memory_scale=0.25,
            attnres_final_memory_groups=1,
            attnres_final_memory_bigram_buckets=64,
            attnres_final_memory_bigram_banks=2,
            attnres_final_memory_gate_cap=0.85,
            attnres_final_memory_gate_bias_init=4.0,
        )
        model = train.GPT(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))
        loss = model(idx, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('final_memory_query_gate', model.last_attnres_metrics)
        self.assertIn('final_memory_query_norm', model.last_attnres_metrics)
        self.assertNotIn('memory_total', model.last_attnres_metrics)

    def test_query_projected_final_memory_forward(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=16,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            window_pattern='L',
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=0,
            attnres_final_memory_mode='query_projected',
            attnres_final_memory_source='tied_bigram_only',
            attnres_final_memory_value_mode='rmsnorm',
            attnres_final_memory_scale=0.25,
            attnres_final_memory_groups=1,
            attnres_final_memory_bigram_buckets=64,
            attnres_final_memory_bigram_banks=2,
            attnres_final_memory_gate_cap=0.85,
            attnres_final_memory_gate_bias_init=4.0,
        )
        model = train.GPT(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))
        loss = model(idx, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.attnres_final_memory_gate_proj)
        self.assertIn('final_memory_query_delta_norm', model.last_attnres_metrics)

    def test_token_register_scale_warmup_changes_runtime_scale(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=1,
            attnres_token_register_mode='query_local',
            attnres_token_register_bias_mode='static',
            attnres_token_register_value_mode='raw',
            attnres_token_register_scale=0.3125,
            attnres_token_register_scale_start=0.0,
            attnres_token_register_scale_warmup_steps=10,
        )
        model = train.GPT(cfg)
        model.init_weights()
        model.attnres_token_register_embeds[0][0].weight.data.fill_(1.0)
        idx = torch.tensor([[1, 2]], dtype=torch.long)

        model.set_attnres_runtime_step(0)
        regs0, _ = model._build_attnres_token_registers(idx, query_index=0, bias_input=None)
        self.assertTrue(torch.allclose(regs0, torch.zeros_like(regs0)))

        model.set_attnres_runtime_step(5)
        regs5, _ = model._build_attnres_token_registers(idx, query_index=0, bias_input=None)
        self.assertTrue(torch.allclose(regs5, torch.full_like(regs5, 0.15625)))

        model.set_attnres_runtime_step(10)
        regs10, _ = model._build_attnres_token_registers(idx, query_index=0, bias_input=None)
        self.assertTrue(torch.allclose(regs10, torch.full_like(regs10, 0.3125)))

    def test_extra_register_bias_broadcast_matches_scalar_bias(self):
        mixer = train.DepthSoftmaxMixer(num_queries=1, dim=4, num_registers=0)
        mixer.init_weights()
        source_a = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        source_b = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
        extra = torch.tensor([[[[0.0, 0.0, 1.0, 0.0]]]])
        bias_1d = torch.tensor([1.25])
        bias_3d = bias_1d.view(1, 1, 1)

        mixed_a, weights_a, _ = mixer.mix_with_weights(
            query_index=0,
            source_values=[source_a, source_b],
            include_registers=True,
            extra_registers=extra,
            extra_register_bias=bias_1d,
        )
        mixed_b, weights_b, _ = mixer.mix_with_weights(
            query_index=0,
            source_values=[source_a, source_b],
            include_registers=True,
            extra_registers=extra,
            extra_register_bias=bias_3d,
        )

        self.assertTrue(torch.allclose(weights_a, weights_b, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(mixed_a, mixed_b, atol=1e-7, rtol=0.0))

    def test_zero_projected_bias_matches_static_bias(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=1,
            attnres_token_register_mode='query_local',
            attnres_token_register_bias_mode='projected',
            attnres_token_register_value_mode='rmsnorm',
            attnres_token_register_scale=1.0,
        )
        model = train.GPT(cfg)
        model.init_weights()
        for proj in model.attnres_token_register_bias_proj:
            proj.weight.data.zero_()
        model.attnres_token_register_bias.data.fill_(-5.25)

        idx = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        bias_input = torch.randn(2, 3, cfg.n_embd)
        _, extra_bias = model._build_attnres_token_registers(idx, query_index=0, bias_input=bias_input)

        expected = torch.full((1, idx.size(0), idx.size(1)), -5.25, dtype=extra_bias.dtype)
        self.assertEqual(tuple(extra_bias.shape), (1, idx.size(0), idx.size(1)))
        self.assertTrue(torch.allclose(extra_bias, expected, atol=1e-7, rtol=0.0))

    def test_projected_delta_scale_scales_dynamic_component(self):
        common = dict(
            sequence_len=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=1,
            attnres_token_register_mode='query_local',
            attnres_token_register_bias_mode='projected',
            attnres_token_register_value_mode='raw',
            attnres_token_register_scale=1.0,
        )
        model_a = train.GPT(train.GPTConfig(**common, attnres_token_register_delta_scale=1.0))
        model_b = train.GPT(train.GPTConfig(**common, attnres_token_register_delta_scale=0.5))
        model_a.init_weights()
        model_b.init_weights()
        model_b.load_state_dict(model_a.state_dict())
        for proj in model_a.attnres_token_register_bias_proj:
            proj.weight.data.fill_(1.0)
        for proj in model_b.attnres_token_register_bias_proj:
            proj.weight.data.fill_(1.0)
        model_a.attnres_token_register_bias.data.zero_()
        model_b.attnres_token_register_bias.data.zero_()
        idx = torch.tensor([[1, 2]], dtype=torch.long)
        bias_input = torch.ones(1, 2, 8)

        _, bias_a = model_a._build_attnres_token_registers(idx, query_index=0, bias_input=bias_input)
        _, bias_b = model_b._build_attnres_token_registers(idx, query_index=0, bias_input=bias_input)
        self.assertTrue(torch.allclose(bias_b, bias_a * 0.5, atol=1e-6, rtol=0.0))

    def test_query_local_token_registers_use_query_specific_table(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=1,
            attnres_token_register_mode='query_local',
            attnres_token_register_bias_mode='static',
            attnres_token_register_value_mode='raw',
            attnres_token_register_scale=1.0,
        )
        model = train.GPT(cfg)
        model.init_weights()
        embeds = model.attnres_token_register_embeds[0]
        embeds[0].weight.data.fill_(1.0)
        embeds[1].weight.data.fill_(2.0)

        idx = torch.tensor([[1, 2]], dtype=torch.long)
        regs0, _ = model._build_attnres_token_registers(idx, query_index=0, bias_input=None)
        regs1, _ = model._build_attnres_token_registers(idx, query_index=1, bias_input=None)

        self.assertTrue(torch.allclose(regs0, torch.ones_like(regs0)))
        self.assertTrue(torch.allclose(regs1, torch.full_like(regs1, 2.0)))
        self.assertFalse(torch.allclose(regs0, regs1))

    def test_extra_register_weight_cap_bounds_token_register_mass(self):
        mixer = train.DepthSoftmaxMixer(num_queries=1, dim=4, num_registers=0)
        mixer.init_weights()
        source_a = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        source_b = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
        extra = torch.tensor([[[[0.0, 0.0, 1.0, 0.0]]]])
        _, weights, _ = mixer.mix_with_weights(
            query_index=0,
            source_values=[source_a, source_b],
            include_registers=True,
            extra_registers=extra,
            extra_register_bias=torch.tensor([10.0]),
            extra_register_weight_cap=0.3,
        )
        self.assertLessEqual(float(weights[-1].item()), 0.300001)

    def test_final_only_scope_only_enables_final_registers(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            attnres_mode='block',
            attnres_block_size=2,
            attnres_weight_mode='softmax',
            attnres_num_registers=0,
            attnres_token_registers=1,
            attnres_token_register_mode='query_local',
            attnres_token_register_bias_mode='projected',
            attnres_token_register_value_mode='rmsnorm',
            attnres_token_register_scale=0.3125,
            attnres_register_scope='final_only',
        )
        model = train.GPT(cfg)
        self.assertFalse(model._attnres_use_registers("attn_only"))
        self.assertFalse(model._attnres_use_registers("mlp_only"))
        self.assertTrue(model._attnres_use_registers("final_only"))

    def test_sample_logit_grads_rows_sum_to_zero(self):
        logits = torch.tensor([[[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]]], dtype=torch.float32)
        targets = torch.tensor([[0, 1]], dtype=torch.long)
        grads = train.sample_logit_grads(
            logits,
            targets,
            max_positions=8,
            ignore_index=-1,
            normalize_rows=False,
        )
        self.assertEqual(tuple(grads.shape), (2, 3))
        self.assertTrue(torch.allclose(grads.sum(dim=-1), torch.zeros(2), atol=1e-6, rtol=0.0))

    def test_lm_head_lost_fraction_is_zero_when_weight_spans_vocab(self):
        grads = torch.tensor([[1.0, -1.0], [-0.5, 0.5]], dtype=torch.float32)
        weight = torch.eye(2, dtype=torch.float32)
        lost = train.lm_head_lost_fraction(grads, weight)
        self.assertAlmostEqual(lost, 0.0, places=6)

    def test_rotate_lm_head_reduces_lost_fraction_on_synthetic_signal(self):
        lm_head = torch.nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            lm_head.weight.copy_(torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]))
        grads = torch.tensor([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ], dtype=torch.float32)
        before = train.lm_head_lost_fraction(grads, lm_head.weight.detach())
        stats = train.rotate_lm_head_towards_lost_dirs_(lm_head, grads, rank=1, alpha=1.0)
        after = train.lm_head_lost_fraction(grads, lm_head.weight.detach())
        self.assertIsNotNone(stats)
        self.assertGreater(before, 0.0)
        self.assertLess(after, before)

    def test_lm_head_rotation_requires_untied_head(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            tie_lm_head=True,
            lm_head_rotation_every=8,
            attnres_mode='off',
        )
        with self.assertRaises(ValueError):
            train.GPT(cfg)

    def test_maybe_rotate_lm_head_runs_end_to_end(self):
        cfg = train.GPTConfig(
            sequence_len=8,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            use_value_embeds=False,
            use_rotary=False,
            use_qk_norm=False,
            use_absolute_positions=True,
            norm_type='layernorm',
            mlp_activation='gelu',
            reference_init=True,
            reference_optimizer=True,
            tie_lm_head=False,
            lm_head_rotation_every=1,
            lm_head_rotation_max_positions=4,
            lm_head_rotation_buffer_rows=8,
            lm_head_rotation_rank=2,
            lm_head_rotation_alpha=0.1,
            attnres_mode='off',
        )
        model = train.GPT(cfg)
        model.init_weights()
        logits = torch.randn(1, 3, 16)
        targets = torch.tensor([[2, 3, 4]], dtype=torch.long)
        model._observe_lm_head_rotation(logits, targets)
        self.assertGreater(model.lm_head_rotation_buffer.n_rows, 0)
        stats = model.maybe_rotate_lm_head(step=1, optimizer=None)
        self.assertIsNotNone(stats)
        self.assertEqual(model.lm_head_rotation_updates, 1)

    def test_parse_required_summary_metrics_rejects_incomplete_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.log"
            bad.write_text("val_bpb: 2.0\n")
            with self.assertRaises(ValueError):
                parse_required_summary_metrics(bad)


if __name__ == '__main__':
    unittest.main()
