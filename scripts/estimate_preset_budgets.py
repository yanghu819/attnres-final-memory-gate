from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from autoresearch_attnres_project.presets import PRESETS
DEFAULT_PRESETS = [
    'attnres_block2',
    'final_memory_tied_bigram_factorized_r128_b524288_c085',
    'input_bigram_hash_smear',
    'golf_attnres_tiny',
    'golf_input_bigram_smear_tiny',
    'golf_final_lightblend_tiny',
]

CHILD = r'''
import json
import os
import torch
from autoresearch_attnres_project.legacy_engine import build_model_config, GPT, DEPTH

vocab_size = int(os.environ.get('AUTORESEARCH_VOCAB_SIZE', '4096'))
config = build_model_config(DEPTH, vocab_size)
with torch.device('meta'):
    model = GPT(config)
counts = model.num_scaling_params()
out = {
    'model_dim': config.n_embd,
    'n_layer': config.n_layer,
    'vocab_size': config.vocab_size,
    'param_counts': counts,
}
print(json.dumps(out))
'''


def estimate_one(name: str) -> dict[str, object]:
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    for key, value in PRESETS[name].env.items():
        env.setdefault(key, value)
    proc = subprocess.run(
        [sys.executable, '-c', CHILD],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(proc.stdout)
    total = int(data['param_counts']['total'])
    artifact_mb = {
        'fp16': total * 2 / (1024 * 1024),
        'int8': total * 1 / (1024 * 1024),
        'int6': total * 0.75 / (1024 * 1024),
        'int5': total * (5 / 8) / (1024 * 1024),
        'int4': total * 0.5 / (1024 * 1024),
    }
    return {
        'preset': name,
        'model_dim': data['model_dim'],
        'n_layer': data['n_layer'],
        'vocab_size': data['vocab_size'],
        'param_counts': data['param_counts'],
        'artifact_mb': artifact_mb,
    }


def main(argv: list[str]) -> int:
    presets = argv[1:] if len(argv) > 1 else DEFAULT_PRESETS
    rows = [estimate_one(name) for name in presets]
    print('preset\ttotal_params\tattnres_params\tfp16_mb\tint8_mb\tint5_mb\tint4_mb\tmodel_dim\tlayers')
    for row in rows:
        counts = row['param_counts']
        art = row['artifact_mb']
        print(
            f"{row['preset']}\t{counts['total']}\t{counts['attnres']}\t"
            f"{art['fp16']:.2f}\t{art['int8']:.2f}\t{art['int5']:.2f}\t{art['int4']:.2f}\t"
            f"{row['model_dim']}\t{row['n_layer']}"
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
