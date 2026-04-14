# 显存与内存占用估算

## 概览

```
总显存需求 = 权重显存 + KV cache pool + Mamba cache（混合模型）+ 激活/workspace 预留
```

推理时激活值占比小，SGLang 自动预留；主要关注前三项。

---

## 1. 权重显存（最大头）

### 公式

```
weight_vram_total  = 参数量(B) × 每参数字节数   (GB)
weight_vram_per_gpu = weight_vram_total / tp_size
```

**口诀：BF16 模型 ≈ 参数量(B) × 2 GB**

### 数据类型对照

| 精度 | 字节/参数 | 70B 模型权重 | 备注 |
|------|-----------|-------------|------|
| FP32 | 4 | 280 GB | 训练用 |
| BF16 / FP16 | 2 | 140 GB | 推理主流 |
| FP8 | 1 | 70 GB | 新兴量化 |
| INT8 | 1 | 70 GB | 量化 |
| INT4 / GPTQ / AWQ | ~0.5 | ~35 GB | 激进量化 |
| GGUF Q4_K_M | ~0.45 | ~32 GB | llama.cpp 格式 |

### 从 HF 快速读取参数量

**方法一（最快）：看 Files 页面文件总大小**

HF → Files and versions → 所有 `.safetensors` 文件大小之和 ≈ BF16 精度的权重显存需求。

例：Qwen3.5-9B 文件合计 ~18 GB → BF16 权重需 ~18 GB 显存。

**方法二：从 config.json 精确计算**

```python
# 简化公式（忽略 embedding 重叠时）
attn = hidden × (num_heads + 2×kv_heads) × head_dim × 2  # QKV + O proj (每层)
ffn  = hidden × intermediate × 3                          # SwiGLU gate/up/down (每层)
emb  = vocab_size × hidden × 2                           # in/out embedding

total_params = num_layers × (attn + ffn) + emb
total_gb = total_params × 2 / 1e9  # BF16
```

Qwen3.5-9B 实测（已验证）：
```
hidden=4096, num_heads=16, kv_heads=4, head_dim=256, layers=32
intermediate=12288, vocab=248320

attn/层 = 4096×(16+8)×256×2 = 50,331,648
ffn/层  = 4096×12288×3     = 150,994,944
emb     = 248320×4096×2     = 2,034,237,440

total ≈ 32×(50.3M+151.0M) + 2034.2M = 6,444M + 2,034M ≈ 7.2B 参数
BF16 权重 = 7.2B × 2 = 14.4 GB（单卡 TP=1 下）
```

> 注：SGLang 实测 `mem_usage=8.85 GB`（TP=2，每卡加载一半 = 7.2 GB），与手算吻合。

---

## 2. KV Cache 显存

### 纯 Transformer 模型（Llama、Qwen2.5 等）

```
per_token_kv_bytes = num_attention_layers × 2(K+V) × num_kv_heads × head_dim × bytes_per_elem

total_kv = per_token_kv_bytes × max_total_num_tokens  # SGLang 的 KV pool 大小
```

### 混合架构（Qwen3.5、Jamba 等）

只有少数 Full Attention 层有 KV cache，其余 Mamba 层用 SSM state。

Qwen3.5-9B（32层中有 8 层是 Full Attention，由 config.json full_attention_interval=4 确认）：
```
# TP=1 时（不分片）：
per_token_kv = 8 × 2 × 4 × 256 × 2 = 32,768 bytes = 32 KB/token

# TP=2 时（kv_heads 4 → 每卡 2）：
per_token_kv_per_gpu = 8 × 2 × 2 × 256 × 2 = 16,384 bytes = 16 KB/token
→ 5 GB KV pool × 1024³ / 16384 = 327K tokens ✓（与日志一致）
```

此外还有 **Mamba cache**（固定大小，不随 context 增长）：
```
mamba_cache = max_concurrent_requests × per_request_ssm_state
```
Qwen3.5-9B 实测：186 并发 × ~23.5 MB = **4.48 GB**

### SGLang KV cache pool 精确计算公式

来自源码 `model_runner_kv_cache_mixin.py:65`：

```python
rest_memory = post_load_avail - pre_load_avail × (1 - mem_fraction_static)
# = 加载后剩余内存 - 激活/workspace 预留
```

Qwen3.5-9B 实测代入（GPU 0/2，各 ~24 GB）：
```
pre_load_avail  = 22.93 GB
post_load_avail = 14.08 GB  (加载 8.85 GB 权重后)
mem_fraction_static = 0.8

激活预留 = 22.93 × 0.2  = 4.59 GB
rest     = 14.08 - 4.59 = 9.49 GB  (Mamba + KV 共享)

Mamba = 9.49 × 0.9/1.9 = 4.48 GB  → max 186 并发
KV    = 9.49 × 1.0/1.9 = 5.00 GB  → 327,310 tokens ✓
```

> `mamba_full_memory_ratio=0.9`（`server_args.py:561`）为默认值，可调。

---

## 3. CPU 内存（DRAM）需求

HuggingFace 加载权重时，**先全部加载到 CPU RAM，再转移 GPU**。

```
CPU RAM 峰值 ≈ 模型权重总大小 × 1.2 + 系统开销(~4 GB)
```

| 模型 | 精度 | CPU RAM 需求 |
|------|------|-------------|
| 7B | BF16 | ~18 GB |
| 32B | BF16 | ~77 GB |
| 70B | BF16 | ~168 GB（超多数服务器！） |
| 70B | INT4 | ~42 GB |

**检查方法**：
```bash
free -h          # 查看 CPU RAM 总量和可用量
cat /proc/meminfo | grep MemTotal
```

---

## 4. 快速判断能否跑起来

### 三个条件

```
① 每卡权重显存 < 每卡总显存 × 0.75   (留 25% 给 KV + 激活)
② 总权重 × 1.2 < CPU RAM 总量
③ 所需 context 对应的 KV pool ≤ 剩余显存 × 0.8
```

### 5× RTX 4090，24 GB × 5 = 120 GB 能跑什么

| 模型 | 精度 | 总权重 | 最小 TP | 每卡权重 | 能跑？ |
|------|------|--------|---------|---------|--------|
| 7B | BF16 | 14 GB | TP=1 | 14 GB | ✓（剩 10 GB → KV） |
| 9B | BF16 | 18 GB | TP=1 | 18 GB | ✓（略紧） |
| 14B | BF16 | 28 GB | TP=2 | 14 GB | ✓ |
| 32B | BF16 | 64 GB | TP=4 | 16 GB | ✓（剩 8 GB → KV） |
| 72B | BF16 | 144 GB | - | >24 GB | ✗（单卡超了） |
| 72B | INT4 | 36 GB | TP=2 | 18 GB | ✓ |
| 70B | FP8 | 70 GB | TP=5 | 14 GB | ✓ |
| 236B MoE | BF16 | 472 GB | - | - | ✗ |
| 236B MoE | FP8 | 236 GB | TP=5 | 47 GB | ✗（单卡超了） |

### 快速估算脚本

[../scripts/canrun.py](../scripts/canrun.py)

---

## 5. MoE 模型的特殊情况

MoE 模型（Mixtral、DeepSeek-V2/V3 等）的参数量分"总参数"和"激活参数"：

```
显存需要装下 全部参数（所有 Expert 的权重都要在显存里）
计算只用到   top-K Expert 的激活参数
```

所以：
- Mixtral 8×7B（总 46B）≠ 7B，需要 ~92 GB 显存（BF16）
- DeepSeek-V2（总 236B，激活 21B）需要 ~472 GB 显存（BF16），但推理算力只相当于 21B

EP（Expert Parallelism）可以把不同 Expert 分到不同 GPU，降低单卡需求（见 topics/02-moe.md）。

---

## 结论/经验

1. **先看 HF Files 总大小**：秒级判断，BF16 文件大小 ≈ 权重显存需求
2. **主要瓶颈通常是显存而非 CPU RAM**（14B 以下模型）；70B+ BF16 可能 CPU RAM 先不够
3. **量化是突破显存限制的关键**：INT4 把 72B 从 144 GB 压到 36 GB
4. **混合架构（Mamba）KV cache 极小**：Qwen3.5-9B 的 KV pool 只有 5 GB，但支持 327K tokens
5. **TP 越大，每卡权重越小，KV pool 越大**（同等显存下支持更长上下文）
