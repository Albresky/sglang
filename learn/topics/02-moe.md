# MoE（Mixture of Experts）

## 核心思想

传统 Dense 模型：每个 Token 经过**所有** FFN 层的参数。
MoE 模型：每个 Token 只经过 FFN 层中**被选中的少数 Expert**（稀疏激活）。

```
输入 Token
    ↓
  Router（门控网络）
    ↓  ↓  ↓
 E1  E2  E3  E4  E5  E6  E7  E8  ← 8 个 Expert（FFN 块）
 ↑        ↑                       ← Router 选 top-2：E1 + E3
    ↓
加权求和输出
```

---

## 关键参数

| 概念 | 说明 |
|------|------|
| `num_experts` | Expert 总数，如 64 |
| `top_k` | 每个 Token 激活的 Expert 数，通常 2 |
| `activated_params` | 实际参与计算的参数量 = 总参数 × (top_k / num_experts) |
| `routed_scaling_factor` | 路由权重缩放系数 |

### 典型 MoE 模型对比

| 模型 | 总参数 | Expert 数 | Top-K | 激活参数 |
|------|--------|-----------|-------|----------|
| Qwen2-57B-A14B | 57B | 64 | 2 | ~14B |
| DeepSeek-V2 | 236B | 160 | 6 | ~21B |
| Mixtral-8×7B | 47B | 8 | 2 | ~13B |
| DeepSeek-V3 | 671B | 256 | 8 | ~37B |

---

## 显存影响

MoE 模型的**计算量**接近激活参数，但**权重显存**需要装下所有 Expert：
- Mixtral-8×7B：激活参数 ~13B，但权重显存需要按 ~47B 计算
- 因此 MoE 显存大，但推理速度（FLOPs）接近小模型

---

## Router 机制

### Top-K Gating（最常见）
```python
# 伪代码
logits = router_linear(hidden_state)          # [seq_len, num_experts]
top_k_weights, top_k_indices = topk(logits, k=2)
weights = softmax(top_k_weights)              # 归一化
output = sum(weights[i] * expert[i](x) for i in top_k_indices)
```

### Load Balancing
Router 有天然的不平衡风险（某个 Expert 被过度选中）。
解决方案：训练时加 **auxiliary loss**（辅助负载均衡损失）。

---

## Expert Parallelism 与 MoE 的关系

EP 是专门为 MoE 设计的并行策略：
- 将不同 Expert 放到不同 GPU
- Token 通过 `All2All` 发送到对应 Expert 的 GPU
- 每张 GPU 只需存 `num_experts / ep_size` 个 Expert

```
GPU 0: Expert 0~15
GPU 1: Expert 16~31
GPU 2: Expert 32~47
GPU 3: Expert 48~63

Token A → Router → Expert 7 → 发到 GPU 0 计算
Token B → Router → Expert 42 → 发到 GPU 2 计算
```

---

## SGLang 中的 MoE 支持

```bash
# 启动 MoE 模型（以 Qwen2-57B-A14B 为例，需要足够显存）
python -m sglang.launch_server \
    --model-path Qwen/Qwen2-57B-A14B-Instruct \
    --tp-size 4 \
    --port 9991

# 如果 GPU 够多，可以用 EP
python -m sglang.launch_server \
    --model-path Qwen/Qwen2-57B-A14B-Instruct \
    --tp-size 2 --ep-size 4 \
    --port 9991
```

### 源码位置
- `python/sglang/srt/models/qwen2_moe.py` — Qwen2 MoE 实现
- `python/sglang/srt/layers/moe/` — Expert 路由与计算核心

---

## 动手实验建议

Qwen3.5-9B 是 Dense 模型，无法演示 MoE。待换用 MoE 模型后记录实验。

## 结论/心得

> 待填写
