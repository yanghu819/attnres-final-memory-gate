from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    env: Dict[str, str] = field(default_factory=dict)


def _common_reference_env() -> Dict[str, str]:
    return {
        "AUTORESEARCH_ATTN_BACKEND": "sdpa",
        "AUTORESEARCH_USE_COMPILE": "0",
        "AUTORESEARCH_NANOGPT_REF": "1",
        "AUTORESEARCH_WINDOW_PATTERN": "L",
        "AUTORESEARCH_MAX_SEQ_LEN": "256",
        "AUTORESEARCH_DEPTH": "6",
        "AUTORESEARCH_ASPECT_RATIO": "64",
        "AUTORESEARCH_HEAD_DIM": "64",
        "AUTORESEARCH_VOCAB_SIZE": "4096",
        "AUTORESEARCH_DEVICE_BATCH_SIZE": "64",
        "AUTORESEARCH_TOTAL_BATCH_SIZE": "16384",
        "AUTORESEARCH_REFERENCE_DROPOUT": "0.2",
        "AUTORESEARCH_REFERENCE_BIAS": "0",
        "AUTORESEARCH_DIFFATTN_MODE": "off",
        "AUTORESEARCH_PERMIX_MODE": "off",
        "AUTORESEARCH_SEGATTN_MODE": "off",
        "AUTORESEARCH_MODA_MODE": "off",
        "AUTORESEARCH_ANCHORKV_MODE": "off",
    }


BASELINE_ENV = {
    **_common_reference_env(),
    "AUTORESEARCH_ATTNRES_MODE": "off",
    "AUTORESEARCH_ATTNRES_NUM_REGISTERS": "0",
    "AUTORESEARCH_ATTNRES_TOKEN_REGISTERS": "0",
}

ATTNRES_BLOCK2_ENV = {
    **_common_reference_env(),
    "AUTORESEARCH_ATTNRES_MODE": "block",
    "AUTORESEARCH_ATTNRES_BLOCK_SIZE": "2",
    "AUTORESEARCH_ATTNRES_WEIGHT_MODE": "softmax",
    "AUTORESEARCH_ATTNRES_NUM_REGISTERS": "0",
    "AUTORESEARCH_ATTNRES_TOKEN_REGISTERS": "0",
    "AUTORESEARCH_ATTNRES_REGISTER_SCOPE": "all",
}

PROJECTED_COMMON_ENV = {
    **ATTNRES_BLOCK2_ENV,
    "AUTORESEARCH_ATTNRES_TOKEN_REGISTERS": "1",
    "AUTORESEARCH_ATTNRES_TOKEN_REGISTER_MODE": "query_local",
    "AUTORESEARCH_ATTNRES_TOKEN_REGISTER_BIAS_MODE": "projected",
    "AUTORESEARCH_ATTNRES_TOKEN_REGISTER_VALUE_MODE": "rmsnorm",
    "AUTORESEARCH_ATTNRES_TOKEN_QUERY_MODE": "off",
    "AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SPREAD_COEF": "0.0",
    "AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SPREAD_SAMPLES": "128",
    "AUTORESEARCH_ATTNRES_REGISTER_BIAS_INIT": "-5.25",
    "AUTORESEARCH_ATTNRES_REGISTER_BIAS_START": "-8.0",
    "AUTORESEARCH_ATTNRES_REGISTER_BIAS_TARGET": "-5.25",
    "AUTORESEARCH_ATTNRES_REGISTER_BIAS_WARMUP_STEPS": "64",
    "AUTORESEARCH_ATTNRES_TARGET_REGISTER_LAYER_MIN": "0.003",
    "AUTORESEARCH_ATTNRES_TARGET_REGISTER_LAYER_MAX": "0.020",
    "AUTORESEARCH_ATTNRES_TARGET_REGISTER_FINAL_MIN": "0.001",
    "AUTORESEARCH_ATTNRES_TARGET_REGISTER_FINAL_MAX": "0.010",
    "AUTORESEARCH_ATTNRES_TARGET_REGISTER_COEF": "0.020",
}

FINAL_MEMORY_COMMON_ENV = {
    **ATTNRES_BLOCK2_ENV,
    "AUTORESEARCH_ATTNRES_FINAL_MEMORY_VALUE_MODE": "rmsnorm",
    "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE": "full",
    "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE": "0.3125",
    "AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP": "0.85",
    "AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT": "4.0",
}

INPUT_MEMORY_COMMON_ENV = {
    **ATTNRES_BLOCK2_ENV,
    "AUTORESEARCH_ATTNRES_INPUT_MEMORY_VALUE_MODE": "rmsnorm",
    "AUTORESEARCH_ATTNRES_INPUT_MEMORY_SCALE": "1.0",
    "AUTORESEARCH_ATTNRES_INPUT_MEMORY_HASH_DIM": "64",
    "AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_BUCKETS": "16384",
    "AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_BANKS": "4",
    "AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_SCALE": "1.0",
    "AUTORESEARCH_ATTNRES_INPUT_MEMORY_TRIGRAM_BUCKETS": "16384",
    "AUTORESEARCH_ATTNRES_INPUT_MEMORY_TRIGRAM_BANKS": "4",
    "AUTORESEARCH_ATTNRES_INPUT_MEMORY_TRIGRAM_SCALE": "1.0",
    "AUTORESEARCH_ATTNRES_INPUT_MEMORY_SMEAR_BIAS_INIT": "-4.0",
}

GOLF_TINY_COMMON_ENV = {
    **ATTNRES_BLOCK2_ENV,
    "AUTORESEARCH_DEPTH": "10",
    "AUTORESEARCH_ASPECT_RATIO": "8",
    "AUTORESEARCH_DEVICE_BATCH_SIZE": "64",
    "AUTORESEARCH_TOTAL_BATCH_SIZE": "16384",
}

PRESETS = {
    "baseline_off": Preset(
        name="baseline_off",
        description="Reference autoresearch backbone with AttnRes disabled.",
        env=BASELINE_ENV,
    ),
    "attnres_block2": Preset(
        name="attnres_block2",
        description="Faithful AttnRes strong baseline: block size 2, softmax, no registers.",
        env=ATTNRES_BLOCK2_ENV,
    ),
    "attnres_block2_untied": Preset(
        name="attnres_block2_untied",
        description="Faithful AttnRes strong baseline with an untied LM head for LM-head rotation ablations.",
        env={
            **ATTNRES_BLOCK2_ENV,
            "AUTORESEARCH_UNTIE_LM_HEAD": "1",
        },
    ),
    "attnres_block2_lmrotate_r16": Preset(
        name="attnres_block2_lmrotate_r16",
        description="AttnRes strong baseline with untied LM head and periodic gradient-aware LM-head rotation.",
        env={
            **ATTNRES_BLOCK2_ENV,
            "AUTORESEARCH_UNTIE_LM_HEAD": "1",
            "AUTORESEARCH_LM_HEAD_ROTATE_EVERY": "64",
            "AUTORESEARCH_LM_HEAD_ROTATE_RANK": "16",
            "AUTORESEARCH_LM_HEAD_ROTATE_ALPHA": "0.05",
            "AUTORESEARCH_LM_HEAD_ROTATE_MAX_POSITIONS": "64",
            "AUTORESEARCH_LM_HEAD_ROTATE_BUFFER_ROWS": "2048",
            "AUTORESEARCH_LM_HEAD_ROTATE_BUFFER_DEVICE": "cpu",
        },
    ),
    "projected_deepemb_final": Preset(
        name="projected_deepemb_final",
        description="Projected token-memory register on top of AttnRes block2, enabled at final mixer only.",
        env={
            **PROJECTED_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_REGISTER_SCOPE": "final_only",
            "AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SCALE": "0.3125",
        },
    ),
    "projected_deepemb_final_static": Preset(
        name="projected_deepemb_final_static",
        description="Static-bias token-memory register on top of AttnRes block2, enabled at final mixer only.",
        env={
            **PROJECTED_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_REGISTER_SCOPE": "final_only",
            "AUTORESEARCH_ATTNRES_TOKEN_REGISTER_BIAS_MODE": "static",
            "AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SCALE": "0.3125",
        },
    ),
    "projected_deepemb_all": Preset(
        name="projected_deepemb_all",
        description="Projected token-memory register on top of AttnRes block2, enabled at all AttnRes injection points.",
        env={
            **PROJECTED_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_REGISTER_SCOPE": "all",
            "AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SCALE": "0.25",
        },
    ),
    "final_memory_static": Preset(
        name="final_memory_static",
        description="Minimal bounded final token memory on top of AttnRes block2 with a static scalar gate.",
        env={
            **FINAL_MEMORY_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE": "static",
        },
    ),
    "final_memory_dynamic": Preset(
        name="final_memory_dynamic",
        description="Minimal bounded final token memory on top of AttnRes block2 with a dynamic scalar gate.",
        env={
            **FINAL_MEMORY_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE": "dynamic",
        },
    ),
    "final_memory_factorized_r64": Preset(
        name="final_memory_factorized_r64",
        description="Factorized final token memory on top of AttnRes block2 with rank-64 token memory.",
        env={
            **FINAL_MEMORY_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE": "static",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE": "factorized",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_RANK": "64",
        },
    ),
    "final_memory_tied": Preset(
        name="final_memory_tied",
        description="Tied final token memory that reuses the input embedding table.",
        env={
            **FINAL_MEMORY_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE": "static",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE": "tied",
        },
    ),
    "final_memory_tied_residual_r64": Preset(
        name="final_memory_tied_residual_r64",
        description="Tied final token memory plus a low-rank residual token table.",
        env={
            **FINAL_MEMORY_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE": "static",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE": "tied_residual",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_RESIDUAL_RANK": "64",
        },
    ),
    "final_memory_tied_bigram_b8192": Preset(
        name="final_memory_tied_bigram_b8192",
        description="Tied final token memory plus a hashed bigram residual table.",
        env={
            **FINAL_MEMORY_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE": "static",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE": "tied_bigram",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS": "8192",
        },
    ),
    "final_memory_tied_bigram_factorized_r64_b65536": Preset(
        name="final_memory_tied_bigram_factorized_r64_b65536",
        description="Tied final token memory plus a low-rank hashed bigram residual code projected to hidden size.",
        env={
            **FINAL_MEMORY_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE": "static",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE": "tied_bigram_factorized",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_RANK": "64",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS": "65536",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS": "4",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE": "0.25",
        },
    ),
    "final_memory_tied_bigram_factorized_r128_b262144_c080": Preset(
        name="final_memory_tied_bigram_factorized_r128_b262144_c080",
        description="Low-rank final bigram blend frontier point: tied unigram plus rank-128 hashed bigram code, 256k buckets, cap 0.80.",
        env={
            **FINAL_MEMORY_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE": "static",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE": "tied_bigram_factorized",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_RANK": "128",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS": "262144",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS": "4",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE": "0.25",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP": "0.80",
        },
    ),
    "final_memory_tied_bigram_factorized_r128_b524288_c085": Preset(
        name="final_memory_tied_bigram_factorized_r128_b524288_c085",
        description="Best scalable final-memory blend: tied unigram plus rank-128 hashed bigram code, 512k buckets, cap 0.85.",
        env={
            **FINAL_MEMORY_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE": "static",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE": "tied_bigram_factorized",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_RANK": "128",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS": "524288",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS": "4",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE": "0.25",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP": "0.85",
        },
    ),
    "final_memory_vres_tied": Preset(
        name="final_memory_vres_tied",
        description="Organic value-residual AttnRes: use tied input embedding as memory code and project only into final depth values.",
        env={
            **ATTNRES_BLOCK2_ENV,
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE": "vres",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE": "tied",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_VALUE_MODE": "rmsnorm",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE": "1.0",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_VRES_SCALE": "0.02",
        },
    ),
    "final_memory_modkv": Preset(
        name="final_memory_modkv",
        description="Organic memory-conditioned AttnRes: modulate final depth K/V with tied unigram+bigram memory.",
        env={
            **ATTNRES_BLOCK2_ENV,
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE": "modkv",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE": "tied_bigram",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_VALUE_MODE": "rmsnorm",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE": "0.25",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS": "4",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS": "1048576",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_K_MOD_SCALE": "0.05",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_V_MOD_SCALE": "0.05",
        },
    ),
    "final_memory_modqkv": Preset(
        name="final_memory_modqkv",
        description="Organic memory-conditioned AttnRes: modulate final depth Q/K/V with tied unigram+bigram memory.",
        env={
            **ATTNRES_BLOCK2_ENV,
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE": "modqkv",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE": "tied_bigram",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_VALUE_MODE": "rmsnorm",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE": "0.25",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS": "4",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS": "1048576",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_Q_MOD_SCALE": "0.01",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_K_MOD_SCALE": "0.02",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_V_MOD_SCALE": "0.02",
        },
    ),
    "input_bigram_hash": Preset(
        name="input_bigram_hash",
        description="AttnRes block2 with input-side hashed bigram residual features.",
        env={
            **INPUT_MEMORY_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_INPUT_MEMORY_MODE": "bigram",
        },
    ),
    "input_bigram_hash_smear": Preset(
        name="input_bigram_hash_smear",
        description="AttnRes block2 with input-side hashed bigram residual features plus SmearGate.",
        env={
            **INPUT_MEMORY_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_INPUT_MEMORY_MODE": "bigram",
            "AUTORESEARCH_ATTNRES_INPUT_MEMORY_SMEAR": "1",
        },
    ),
    "input_bigram_trigram_hash": Preset(
        name="input_bigram_trigram_hash",
        description="AttnRes block2 with input-side hashed bigram+trigram residual features.",
        env={
            **INPUT_MEMORY_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_INPUT_MEMORY_MODE": "bigram_trigram",
        },
    ),
    "golf_input_bigram_smear_tiny": Preset(
        name="golf_input_bigram_smear_tiny",
        description="Parameter-golf style tiny AttnRes candidate: small input-side bigram hash with light SmearGate.",
        env={
            **GOLF_TINY_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_INPUT_MEMORY_MODE": "bigram",
            "AUTORESEARCH_ATTNRES_INPUT_MEMORY_SMEAR": "1",
            "AUTORESEARCH_ATTNRES_INPUT_MEMORY_SCALE": "0.25",
            "AUTORESEARCH_ATTNRES_INPUT_MEMORY_HASH_DIM": "16",
            "AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_BUCKETS": "8192",
            "AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_BANKS": "2",
            "AUTORESEARCH_ATTNRES_INPUT_MEMORY_BIGRAM_SCALE": "1.0",
            "AUTORESEARCH_ATTNRES_INPUT_MEMORY_TRIGRAM_SCALE": "0.0",
        },
    ),
    "golf_attnres_tiny": Preset(
        name="golf_attnres_tiny",
        description="Tiny AttnRes block2 baseline sized for parameter-golf style budget experiments.",
        env=GOLF_TINY_COMMON_ENV,
    ),
    "golf_final_lightblend_r64_b32768_c095": Preset(
        name="golf_final_lightblend_r64_b32768_c095",
        description="Tiny final-lightblend cheap preset for parameter-golf style experiments: rank-64, 32k buckets, cap 0.95.",
        env={
            **GOLF_TINY_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE": "static",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE": "tied_bigram_factorized",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_VALUE_MODE": "rmsnorm",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE": "0.25",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP": "0.95",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT": "4.0",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_RANK": "64",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS": "32768",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS": "2",
        },
    ),
    "golf_final_lightblend_r96_b65536_c090": Preset(
        name="golf_final_lightblend_r96_b65536_c090",
        description="Tiny final-lightblend short-budget best for parameter-golf style experiments: rank-96, 64k buckets, cap 0.90.",
        env={
            **GOLF_TINY_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE": "static",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE": "tied_bigram_factorized",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_VALUE_MODE": "rmsnorm",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE": "0.25",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP": "0.90",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT": "4.0",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_RANK": "96",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS": "65536",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS": "2",
        },
    ),
    "golf_final_lightblend_tiny": Preset(
        name="golf_final_lightblend_tiny",
        description="Parameter-golf style tiny AttnRes candidate: bounded final blend with tied unigram and tiny low-rank bigram code.",
        env={
            **GOLF_TINY_COMMON_ENV,
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_MODE": "static",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SOURCE": "tied_bigram_factorized",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_VALUE_MODE": "rmsnorm",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_SCALE": "0.25",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_CAP": "0.85",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_GATE_BIAS_INIT": "4.0",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_RANK": "16",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BUCKETS": "8192",
            "AUTORESEARCH_ATTNRES_FINAL_MEMORY_BIGRAM_BANKS": "2",
        },
    ),
}
