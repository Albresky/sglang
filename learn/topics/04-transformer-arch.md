# Transformer / LLM 架构

## 整体结构

```
输入 Token IDs
    ↓
Embedding Layer（Token → 向量）
    ↓
┌─────────────────────────────┐
│  Transformer Block × N 层   │
│                             │
│  ┌──────────────────────┐   │
│  │  RMSNorm             │   │
│  │  Multi-Head Attention│   │ ← Self-Attention（含 QKV + Output proj）
│  │  残差连接             │   │
│  ├──────────────────────┤   │
│  │  RMSNorm             │   │
│  │  FFN / MoE           │   │ ← 前馈网络（SwiGLU / GeGLU）
│  │  残差连接             │   │
│  └──────────────────────┘   │
└─────────────────────────────┘
    ↓
RMSNorm（最终归一化）
    ↓
LM Head（向量 → Logits → Token）
```

---

## 关键组件详解

### 1. Attention（注意力机制）

```python
# 标准 Multi-Head Attention 伪代码
Q = linear_q(x)   # [batch, seq, num_heads, head_dim]
K = linear_k(x)   # [batch, seq, num_kv_heads, head_dim]  ← GQA 可能更少
V = linear_v(x)

scores = Q @ K.T / sqrt(head_dim)   # 点积注意力
scores = softmax(scores + causal_mask)
output = scores @ V
output = linear_o(output)
```

**GQA（Grouped Query Attention）**：Qwen3 等现代模型使用
- 多个 Q head 共享同一组 K、V head → 减少 KV Cache 显存
- `num_q_heads = 16, num_kv_heads = 8` → 2个Q共享1个KV

**RoPE（Rotary Position Embedding）**：位置编码
- 不需要额外参数，直接在 Q/K 上做旋转变换
- 支持外推到比训练时更长的序列

### 2. FFN（前馈网络）

现代 LLM 常用 **SwiGLU**：
```python
# SwiGLU FFN
gate = silu(linear_gate(x))   # 门控
up = linear_up(x)
output = linear_down(gate * up)

# 参数量：3 × d_model × ffn_dim（3 个矩阵）
```

### 3. RMSNorm（归一化）
```python
# RMSNorm（比 LayerNorm 少一个均值计算步骤）
rms = sqrt(mean(x^2) + eps)
output = x / rms * weight
```

---

## Qwen3.5-9B 架构参数

从 `config.json` 读取（待实验后填入实际值）：
```json
{
  "num_hidden_layers": 36,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 22016,
  "vocab_size": 151936,
  "max_position_embeddings": 131072
}
```

### 参数量验证（估算）
```python
d = 4096        # hidden_size
ffn = 22016     # intermediate_size
vocab = 151936  # vocab_size
L = 36          # num_layers

embedding = vocab * d                        # ~0.6B
attn_per_layer = d * d * 4                  # Q+K+V+O，近似（忽略GQA差异）
ffn_per_layer = d * ffn * 3                 # SwiGLU 3矩阵
total = embedding + L * (attn_per_layer + ffn_per_layer)
print(f"估算参数量: {total/1e9:.1f}B")   # 应接近 9B
```

---

## 推理过程：Prefill vs Decode

### Prefill（预填充）
- 输入：完整的 prompt（N 个 token）
- 操作：一次性并行计算所有 token 的 KV，填满 KV Cache
- 特点：**计算密集**（Compute-bound）

### Decode（逐 token 生成）
- 输入：上一步生成的 1 个 token
- 操作：用新 token 的 Q 与 KV Cache 做 attention
- 特点：**内存密集**（Memory-bound），每步只有 1 个 token 在算

这解释了为什么 LLM 推理的 token/s 会受限于显存带宽而非算力。

---

## 源码对应关系

| 概念 | SGLang 源码位置 |
|------|----------------|
| Qwen3 模型定义 | `python/sglang/srt/models/qwen3.py` |
| Attention 实现 | `python/sglang/srt/layers/attention/` |
| RoPE | `python/sglang/srt/layers/rotary_embedding.py` |
| Radix Cache（KV Cache 管理） | `python/sglang/srt/mem_cache/radix_cache.py` |

---

## 结论/心得

> 待填写
