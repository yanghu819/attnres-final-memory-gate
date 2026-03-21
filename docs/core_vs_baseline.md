# Core Code: Baseline vs Ours

## 1. AttnRes Strong Baseline

The strong baseline is faithful `AttnRes block2`. At the very end, it computes a final depth-mixed output:

```python
# Baseline: final AttnRes output
x, final_weights, final_meta = self.attnres_mixer.final_output_with_weights(
    state,
    include_registers=False,
    extra_registers=None,
    extra_register_bias=None,
    extra_query=None,
    extra_register_weight_cap=0.0,
)
return x
```

Conceptually:

```python
y_attnres = AttnResFinal(x0, h1, h2, ..., hL)
```

There is no token-conditioned memory branch here.

## 2. Our Minimal Method: Static Bounded Final Memory Gate

We do not change the main attention or all-layer routing.
We only add one token-conditioned memory branch at the final output:

```python
# Ours: final-only token memory
memory = self.attnres_final_memory_embed(idx)   # token lookup
memory = rmsnorm(memory)                        # optional normalization
memory = memory * value_scale

gate = torch.sigmoid(gate_bias)                 # scalar gate
gate = gate * gate_cap                          # hard upper bound

y = torch.lerp(y_attnres, memory, gate)
```

Conceptually:

```python
m_token = TokenMemory[token_id]
g = tau * sigmoid(b)
y = (1 - g) * y_attnres + g * m_token
```

Where:

- `y_attnres` is the original `AttnRes block2` final output
- `m_token` is a learned token memory vector
- `tau` is the explicit cap, currently best around `0.85`

This is the current minimal, scalable method.

## 3. Grouped / Multi-Head-Style Extension

If we want a multi-head-style memory without touching the main attention,
we split channels into groups and apply the same gate per group:

```python
B, T, D = x.shape
G = groups
group_dim = D // G

x_groups = x.view(B, T, G, group_dim)
memory_groups = memory.view(B, T, G, group_dim)

gate = torch.sigmoid(group_bias).view(1, 1, G, 1)
gate = gate * gate_cap

mixed_groups = torch.lerp(x_groups, memory_groups, gate)
y = mixed_groups.reshape(B, T, D)
```

Conceptually:

```python
for g in groups:
    y_g = (1 - alpha_g) * y_attnres_g + alpha_g * m_token_g
```

This is the cleanest way to test a "multi-head embedding" idea on top of AttnRes:

- it only changes the final path
- it preserves the AttnRes backbone
- it is easy to ablate with `G = 1 / 2 / 4`

## 4. What Actually Changed

The method is **not** "a more complex AttnRes router".
It is simply:

```python
y = (1 - g) * AttnResFinal(...) + g * TokenMemory[token]
```

And in the grouped version:

```python
y = concat_g((1 - g_g) * AttnResFinal_g(...) + g_g * TokenMemory_g[token])
```

That is the core code difference from baseline.
