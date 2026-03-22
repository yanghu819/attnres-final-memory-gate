# 在 AttnRes 上加入 N-Gram Memory：当前最佳实践

这篇博客只讲现在已经被实验打出来的结论，不沿用早期 `projected token register` 的旧口径。

## 先说结论

我们现在最强的方法，不是把 memory 当成一个新 token 扔进 AttnRes 的 softmax，也不是拿 memory 直接去改 query。

当前最强做法反而更朴素：

```python
m = E_uni[token] + sum_b E_bi_b[hash_b(prev_token, token)]
y = (1 - g) * y_attnres + g * m
```

也就是：

- backbone 还是 faithful `AttnRes block2`
- memory 由 **unigram + hashed bigram** 组成
- 最后通过一个 **bounded gate** 混到 final output

一句白话：

**让 AttnRes 先正常思考，再让 n-gram memory 在最后递一张小抄。**

## 这套 memory 到底是什么

这里的 memory 不是普通小 embedding，而是更像 Engram 风格的哈希记忆：

- `unigram`：当前 token 的记忆向量
- `bigram`：`(prev_token, token)` 的哈希记忆向量
- `bank`：多张独立的 bigram 哈希表

当前 best 用的是：

- `4 banks`
- 每个 bank `1M buckets`

也就是 4 张独立的大 bigram memory 表，查出来后相加。

## 为什么这样用最好

我们真正测过几种更“有机”的接法：

1. memory 直接当 source / KV 进 final softmax  
结论：太容易被 attend 到，会把真实 depth source 挤掉。

2. memory 直接当 query  
结论：太容易把 final routing 推坏，甚至塌成 `x0 = 1.0`。

3. memory 只调制 AttnRes final mixer 的 `q/k/v`  
结论：这是目前最好的有机版本，但还打不过 bounded blend。

所以目前最稳的规律是：

- memory 最适合 **晚用**
- memory 要 **有上限**
- memory 更适合提供 **内容先验**
- memory 不适合直接和真实 depth source 在同一个 softmax 里硬抢

## 现在最强的结果

结果表：

- [`results/ngram_module_ablation_results.tsv`](../results/ngram_module_ablation_results.tsv)
- [`results/bigram_80pct_refine_results.tsv`](../results/bigram_80pct_refine_results.tsv)

当前最强 practical 方法：

- `AttnRes block2` baseline: `2.154041`
- `unigram + bigram, 4 banks x 1M buckets`: `1.840970`

这就是当前最强 best practice。

## 最好的有机版本是什么

如果你坚持“不外挂 branch”，那当前最好的版本是：

- memory 不新增 token
- memory 不新增 source
- memory 只调制 final AttnRes mixer 的 `q/k/v`

也就是 `modqkv`：

```python
q' = q + 0.02 * Wq(memory)
k' = k + 0.05 * Wk(memory)
v' = v + 0.05 * Wv(memory)
```

对应结果：

- [`results/modqkv_refine_results.tsv`](../results/modqkv_refine_results.tsv)

当前 best organic：

- `modqkv q=0.02, k=0.05, v=0.05`: `2.004640`

它明显优于 AttnRes baseline，但还不如上面的 bounded blend。

## 一个重要发现

把 memory 做大，不代表越大越好。

我们测过：

- `4 banks x 1M`
- `5 banks x 1M`
- `6 banks x 1M`
- `7 banks x 1M`

结果是：

- `4x1m` 最好
- 再往上堆没有继续变好

所以问题不是单纯“容量越大越强”，而是：

**memory 的结构要对。**

## 接下来该怎么做

如果继续沿着当前 best practice 往前推，我会优先做：

1. `unigram + bigram + trigram`
2. `tied unigram + bigram residual + trigram residual`
3. 稀疏 bank 选择，而不是更大规模的 dense bank 叠加

也就是继续扩 memory 的**结构化表达**，而不是继续把 AttnRes 路由搞复杂。

## 总结

当前最强方法不是“更复杂的 AttnRes”，而是：

**AttnRes + bounded unigram/bigram memory readout**

而当前最好的有机尝试是：

**memory-conditioned `modqkv`**

但到现在为止，真正打赢所有方案的，仍然是那个更朴素的 final bounded blend。
