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
}
