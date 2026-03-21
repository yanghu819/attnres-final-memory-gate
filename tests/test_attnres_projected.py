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

    def test_parse_required_summary_metrics_rejects_incomplete_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.log"
            bad.write_text("val_bpb: 2.0\n")
            with self.assertRaises(ValueError):
                parse_required_summary_metrics(bad)


if __name__ == '__main__':
    unittest.main()
