# 在 AttnRes 上加入 Projected Token Memory

## 起点

这个项目有两个锚点：

- `autoresearch` 的 reference backbone
- 一个 faithful 的 AttnRes 复现

在这个基础上，我们加了一个 projected token-memory register，放在 `AttnRes block2` 上面。

## 我们真正关心的对比

这个 repo 里有三层 baseline：

1. `baseline_off`
   - 原始 `autoresearch` backbone
2. `attnres_block2`
   - faithful AttnRes strong baseline
3. `projected_deepemb_final`
   - 我们的 canonical 方法

主结果只看这三个 family。当前精确数字以结果表为准：

- `results/main_results.tsv`

## 方法本身是什么

这个 repo 不把方法包装成一个“全层通用的 residual topology 改写”。

我们对外公开的干净版本是：

- AttnRes `block_size=2`
- 一个 token-conditioned register
- projected bias
- `query_local` token table
- `rmsnorm` value mode
- `final_only` routing

一句话：

**AttnRes backbone + final mixer 上的 projected token memory**

## 为什么还要做 ablation

同 cohort 的 ablation 只回答两个问题：

1. `projected` 是否优于 `static`
2. `final_only` 在同一批 seeds 上，是否至少不弱于 `all`

看这里：

- `results/ablation_results.tsv`
- `results/figs/fig_ablation_loss_curves.png`
- `results/figs/fig_ablation_valbpb.png`

## 为什么还要做 fresh-seed follow-up

fresh-seed follow-up 不是主结果，它是稳健性检查。

看这里：

- `results/followup_results.tsv`
- `results/figs/fig_followup_loss_curves.png`
- `results/figs/fig_followup_valbpb.png`

## 一个必须写清楚的 caveat

这个 repo 的公开表述必须老实。

如果最终最优点里 `attnres_final_register` 仍然很高，那么更安全的解释应该是：

- projected token memory 确实有用
- 主要起作用的位置是 final mixer
- 这更像是“叠在 AttnRes 上的 final-path token-memory readout”，而不是一个均匀作用于全层的 routing 改进

这个说法更窄，但也更站得住。
