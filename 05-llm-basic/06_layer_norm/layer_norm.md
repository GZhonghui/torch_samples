# LayerNorm 和 OPT Pre-LN 结构

## 这个项目里的位置

在 `mrl/layers.py` 里，`AttentionLayer` 和 `FFNLayer` 的输入前都会先经过一次
`LayerNorm`，但这两个 `LayerNorm` 不写在 `AttentionLayer.compute()` 或
`FFNLayer.compute()` 内部，而是写在 `TransformerBlock.compute()` 里。

当前 block 的结构是 OPT 的 pre-LN decoder block：

```text
x
 -> LayerNorm
 -> SelfAttention
 -> residual add: x + attn(...)
 -> LayerNorm
 -> FFN
 -> residual add
```

对应代码逻辑可以理解为：

```python
attn_input = layer_norm(hidden, self_attn_norm_weight, self_attn_norm_bias)
hidden = hidden + attention(attn_input)

ffn_input = layer_norm(hidden, final_norm_weight, final_norm_bias)
hidden = hidden + ffn(ffn_input)
```

这里叫 pre-LN，是因为每个子层的归一化发生在子层计算之前：

- `SelfAttention` 之前做 `self_attn_layer_norm`
- `FFN` 之前做 `final_layer_norm`

残差分支上的原始 `hidden` 不先归一化，而是直接和子层输出相加。

## LayerNorm 在做什么

`LayerNorm` 是对每个 token 的 hidden 向量做归一化。

## 什么是归一化

归一化的核心意思是：把一组数值重新缩放到更稳定、更容易计算的分布。

假设某个 token 的 hidden 向量是：

```text
h = [h1, h2, ..., hD]
```

这里 `D = hidden_dim`。LayerNorm 会先在这个向量内部计算均值和方差：

```text
mean = average(h)
var = average((h - mean)^2)
```

然后把每个元素变成：

```text
h_norm = (h - mean) / sqrt(var + eps)
```

这个操作会让这个 hidden 向量大致变成：

```text
均值接近 0
方差接近 1
```

`eps` 是一个很小的数，用来避免 `var` 太小时除以 0。

不过模型不一定希望归一化后的分布永远固定在均值 0、方差 1，所以 LayerNorm
后面还有两个可学习参数：

```text
output = h_norm * weight + bias
```

其中：

```text
weight: [hidden_dim]
bias:   [hidden_dim]
```

这两个参数让模型可以自己学习“归一化之后应该放大多少、平移多少”。所以
LayerNorm 不是简单地把数值强行固定死，而是先稳定尺度，再把合适的尺度交给
模型学习。

如果 hidden shape 是：

```text
[batch_size, seq_len, hidden_dim]
```

那么 `LayerNorm` 沿最后一维 `hidden_dim` 归一化。也就是说，每个 batch、每个
token 位置都有一个长度为 `hidden_dim` 的向量，LayerNorm 会单独处理这个向量。

代码里是：

```python
F.layer_norm(
    hidden,
    (hidden.shape[-1],),
    weight=norm_weight,
    bias=norm_bias,
    eps=1e-5,
)
```

`(hidden.shape[-1],)` 表示只对最后一维做归一化。归一化后还会乘上可学习的
`weight`，再加上可学习的 `bias`。这两个参数的形状都是：

```text
[hidden_dim]
```

直观上，LayerNorm 的作用是让每个 token 的 hidden 向量数值分布更稳定，避免
后面的 attention 或 FFN 直接吃到尺度波动很大的输入。

## Attention 子层的形状

`AttentionLayer.compute()` 的输入是：

```text
x: [batch_size, seq_len, hidden_dim]
```

这个 `x` 已经是经过 `self_attn_norm` 的 hidden。

Q/K/V 投影后仍然是：

```text
q, k, v: [batch_size, seq_len, hidden_dim]
```

然后按 attention head 拆开：

```text
q, k, v: [batch_size, num_heads, seq_len, head_dim]
```

其中：

```text
head_dim = hidden_dim / num_heads
```

attention 计算后会再合并回：

```text
attn_out: [batch_size, seq_len, hidden_dim]
```

最后经过输出投影，返回：

```text
output: [batch_size, seq_len, hidden_dim]
```

所以 Attention 子层整体保持：

```text
[B, S, D] -> [B, S, D]
```

这样它才能和残差分支相加：

```python
hidden = hidden + attention_output
```

## FFN 子层的形状

`FFNLayer.compute()` 的输入是：

```text
x: [batch_size, seq_len, hidden_dim]
```

这个 `x` 已经是经过 `final_norm` 的 hidden。

OPT FFN 是两层线性变换，中间用 ReLU：

```text
fc1: hidden_dim -> 4 * hidden_dim
ReLU
fc2: 4 * hidden_dim -> hidden_dim
```

所以中间形状是：

```text
after fc1: [batch_size, seq_len, 4 * hidden_dim]
```

最终输出回到：

```text
output: [batch_size, seq_len, hidden_dim]
```

所以 FFN 子层整体也保持：

```text
[B, S, D] -> [B, S, D]
```

这样它也可以和残差分支相加：

```python
hidden = hidden + ffn_output
```

## 总结

- `AttentionLayer` 和 `FFNLayer` 的输入前都需要一次 `LayerNorm`。
- 这两个 `LayerNorm` 属于 `TransformerBlock`，不是 attention/FFN 子层自己管理。
- 这是 OPT 的 pre-LN 结构：先归一化，再进入子层，再做残差相加。
- `LayerNorm` 对 `[B, S, D]` 的最后一维 `D` 做归一化。
- Attention 子层保持 `[B, S, D] -> [B, S, D]`。
- FFN 子层也保持 `[B, S, D] -> [B, S, D]`，中间临时扩展到 `4D`。
- 两个子层输出 shape 不变，是为了能和残差分支逐元素相加。
