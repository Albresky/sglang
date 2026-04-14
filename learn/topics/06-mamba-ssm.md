# Mamba / SSM：线性注意力替代方案

## 背景：Attention 的瓶颈

标准 Transformer 的 Self-Attention 每层都需要将当前 token 与所有历史 token 做点积：

```
Attention(Q, K, V) = softmax(QKᵀ/√d) · V
```

- 计算：O(n²)——序列越长越慢
- 内存：每层存 K、V → KV cache 随序列增长

对于 262K token 的长上下文，KV cache 会占用大量显存。

---

## 状态空间模型（SSM）

SSM 来自控制理论，用**固定大小的隐状态 h** 压缩历史：

```
连续形式：
  h'(t) = A h(t) + B x(t)    # 状态更新
  y(t)  = C h(t)             # 输出

离散化后（实际计算）：
  h_t = Ā h_{t-1} + B̄ x_t
  y_t = C h_t
```

其中 A、B、C 是可学习矩阵，h 的大小固定（不随序列增长）。

特性：
- 计算 O(n)，线性复杂度
- 内存 O(1)，状态大小固定
- **无 KV cache**，用 SSM state 替代

---

## Mamba（2023，Gu & Dao）

普通 SSM 的问题：A、B、C 固定，无法根据内容选择性记忆。

Mamba 的改进：**选择性 SSM（Selective SSM）**
- B、C 参数变成输入 x 的函数：`B = B(x_t)`, `C = C(x_t)`
- 模型可以"选择"把哪些信息写入/读出状态
- 类似注意力的 content-aware，但保持 O(n) 复杂度

**Mamba-2**：重新推导为更高效的矩阵乘法形式，训练时支持并行化。

**GDN（Gated Delta Networks）**：Mamba-2 的进一步改进，Qwen3.5 使用此架构。

---

## 混合模型：为什么不用纯 Mamba？

| | 纯 Attention | 纯 Mamba | 混合（Qwen3.5） |
|---|---|---|---|
| 计算复杂度 | O(n²) | O(n) | O(n)≈ |
| 精确 token 回忆 | 好（无损） | 弱（有损压缩） | 好（4 层 Attn） |
| 超长上下文内存 | KV cache 增长 | 固定 | 主要固定 |
| 适用场景 | 精确推理 | 长序列处理 | 两者兼顾 |

Mamba 对"精确复制/回忆特定 token"能力弱，纯 Mamba 在某些推理任务上表现差。
混合模型（少量 Attention + 大量 Mamba）是当前折中方案。

---

## Qwen3.5-9B 的混合架构

从 SGLang 启动日志反推：

```
KV Cache: #tokens=327310, K size=2.50 GB, V size=2.50 GB

# 实验用 TP=2，每卡 KV 存 kv_heads/tp = 4/2 = 2 头
per_token_K_per_gpu = 2.50 GB / 327310 ≈ 8 KB/token

8 KB = num_attn_layers × (kv_heads/tp) × head_dim × 2 bytes(BF16)
     = num_attn_layers × 2 × 256 × 2
→ num_attn_layers = 8  （config.json full_attention_interval=4 确认）
```

**32 层中 8 层 Full Attention（每 4 层一个，full_attention_interval=4），24 层 Mamba/GDN。**

SGLang 日志中的对应信息：
```
# 启动时
Mamba Cache is allocated. max_mamba_cache_size: 186,
  conv_state size: 0.10GB, ssm_state size: 4.38GB

GDN kernel dispatcher: decode=TritonGDNKernel
Using hybrid linear attention backend for hybrid GDN models

# 推理时（每个 decode batch 日志）
mamba num: 2     ← 当前步中活跃的 SSM 状态数（非层数）
mamba usage: 0.05 ← Mamba cache 占用比例
```

---

## SSM State vs KV Cache 对比

| 属性 | KV Cache（Attention层） | SSM State（Mamba层） |
|------|------------------------|---------------------|
| 存储内容 | 原始 K、V 向量（无损） | 压缩后的状态向量（有损） |
| 大小 | 随序列增长（O(n)） | 固定（O(1)） |
| 并发内存 | 每新 token 增加 | 每请求固定大小 |
| Qwen3.5-9B 占用 | 5.00 GB（327K tokens） | 4.48 GB（186 并发请求） |

Qwen3.5-9B 每请求 SSM state 大小：
```
4.38 GB / 186 请求 ≈ 23.5 MB/请求（28 Mamba 层 × 每层状态）
```

---

## 对 SGLang 调度的影响

`max_mamba_cache_size`：同时处理的最大请求数（由 Mamba cache 内存决定，而非 KV cache）。
- 新请求到来时，SGLang 必须为其分配一个固定大小的 SSM state slot
- 如果 Mamba cache 满了，新请求需等待（即使 KV pool 还有空间）

SGLang 参数：
```bash
--mamba-full-memory-ratio 0.9   # Mamba:KV 内存比（默认 0.9:1）
--max-mamba-cache-size N        # 手动限制最大并发 Mamba state 数
```

---

## 学习建议

- 实验：观察长上下文请求（>100K tokens）时 KV pool 是否增长（会），Mamba state 是否增长（不会）
- 实验：同参数量的纯 Transformer 模型（如 Qwen2-7B）KV pool 会大很多，验证架构差异
- 进阶阅读：Mamba 原论文（Gu & Dao 2023），Mamba-2（2024）
