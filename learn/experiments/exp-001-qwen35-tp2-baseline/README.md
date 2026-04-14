# Exp-001: Qwen3.5-9B TP=4 基线启动

## 目标
- 成功启动 `Qwen/Qwen3.5-9B`
- 观察各卡显存分配情况
- 发送测试请求，验证推理正常
- 记录关键日志片段，理解 SGLang 启动过程

## 环境
- GPU：5× RTX 4090（24GB each）
- SGLang 版本：`0.0.0.dev11582+g4df60434d`
- 日期：2026-04-14

---

## Step 1: 记录启动前显存基线

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total \
    --format=csv
```

**输出**：
```
index, memory.used [MiB], memory.free [MiB]
0, 19 MiB, 24063 MiB
1, 8048 MiB, 14507 MiB
2, 18 MiB, 24063 MiB
3, 8042 MiB, 16040 MiB
4, 18 MiB, 24063 MiB
```

---

## Step 2: 启动服务器

```bash
CUDA_VISIBLE_DEVICES=0,2 python -m sglang.launch_server \
    --model-path 'Qwen/Qwen3.5-9B' \
    --port 9991 \
    --tp-size 2 \
    --mem-fraction-static 0.8 \
    --context-length 262144 \
    --reasoning-parser qwen3 \
    --log-level info \
    --log-requests \
    --enable-metrics \
    2>&1 | tee sglang-exp001-enable-metrics.log
```

- `--log-requests`：打印每次请求的 prompt 和完整输出

**关键日志片段**：
```
[2026-04-14 16:08:19] Ignore import error when loading sglang.srt.multimodal.processors.gemma4: cannot import name 'Gemma4AudioConfig' from 'transformers' (/root/wkspace/sglang/.venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-04-14 16:08:32] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/root/wkspace/sglang/.venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-04-14 16:08:32] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/root/wkspace/sglang/.venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-04-14 16:08:32] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/root/wkspace/sglang/.venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-04-14 16:08:32] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/root/wkspace/sglang/.venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-04-14 16:08:32] Using default HuggingFace chat template with detected content format: openai
[2026-04-14 16:08:42 TP0] Ignore import error when loading sglang.srt.multimodal.processors.gemma4: cannot import name 'Gemma4AudioConfig' from 'transformers' (/root/wkspace/sglang/.venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-04-14 16:08:43 TP1] Ignore import error when loading sglang.srt.multimodal.processors.gemma4: cannot import name 'Gemma4AudioConfig' from 'transformers' (/root/wkspace/sglang/.venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-04-14 16:08:43 TP0] Init torch distributed begin.
[2026-04-14 16:08:43 TP1] Init torch distributed begin.
[Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[Gloo] Rank 0 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[Gloo] Rank 0 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[2026-04-14 16:08:43 TP0] sglang is using nccl==2.27.5
[2026-04-14 16:08:44 TP1] Setup Custom allreduce failed with CUDART error: peer access is not supported between these two devices. To silence this warning, specify --disable-custom-all-reduce explicitly.
[2026-04-14 16:08:44 TP0] Setup Custom allreduce failed with CUDART error: peer access is not supported between these two devices. To silence this warning, specify --disable-custom-all-reduce explicitly.
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[2026-04-14 16:08:44 TP1] Init torch distributed ends. elapsed=0.65 s, mem usage=0.19 GB
[2026-04-14 16:08:44 TP0] Init torch distributed ends. elapsed=1.15 s, mem usage=0.19 GB
[2026-04-14 16:08:44 TP1] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/root/wkspace/sglang/.venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-04-14 16:08:44 TP1] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/root/wkspace/sglang/.venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-04-14 16:08:44 TP1] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/root/wkspace/sglang/.venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-04-14 16:08:44 TP1] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/root/wkspace/sglang/.venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-04-14 16:08:44 TP0] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/root/wkspace/sglang/.venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-04-14 16:08:44 TP0] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/root/wkspace/sglang/.venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-04-14 16:08:44 TP0] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/root/wkspace/sglang/.venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-04-14 16:08:44 TP0] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/root/wkspace/sglang/.venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-04-14 16:08:44 TP1] Load weight begin. avail mem=22.93 GB
[2026-04-14 16:08:44 TP0] Load weight begin. avail mem=22.93 GB
[2026-04-14 16:08:44 TP1] Multimodal attention backend not set. Use triton_attn.
[2026-04-14 16:08:44 TP1] Using triton_attn as multimodal attention backend.
[2026-04-14 16:08:44 TP0] Multimodal attention backend not set. Use triton_attn.
[2026-04-14 16:08:44 TP0] Using triton_attn as multimodal attention backend.
`torch_dtype` is deprecated! Use `dtype` instead!
`torch_dtype` is deprecated! Use `dtype` instead!
[2026-04-14 16:08:44 TP1] using attn output gate!
[2026-04-14 16:08:44 TP0] using attn output gate!
[2026-04-14 16:08:45 TP0] Found local HF snapshot for Qwen/Qwen3.5-9B at /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a; skipping download.

Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Multi-thread loading shards:  25% Completed | 1/4 [00:01<00:03,  1.07s/it]
Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.11it/s]
Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.07it/s]
Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.08it/s]
Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.07it/s]
[2026-04-14 16:08:49 TP1] Load weight end. elapsed=5.46 s, type=Qwen3_5ForConditionalGeneration, avail mem=14.08 GB, mem usage=8.85 GB.
[2026-04-14 16:08:49 TP0] Load weight end. elapsed=5.54 s, type=Qwen3_5ForConditionalGeneration, avail mem=14.08 GB, mem usage=8.85 GB.
[2026-04-14 16:08:49 TP0] Using KV cache dtype: torch.bfloat16
[2026-04-14 16:08:49 TP0] Mamba Cache is allocated. max_mamba_cache_size: 186, conv_state size: 0.10GB, ssm_state size: 4.38GB 
[2026-04-14 16:08:49 TP1] Mamba Cache is allocated. max_mamba_cache_size: 186, conv_state size: 0.10GB, ssm_state size: 4.38GB 
[2026-04-14 16:08:49 TP0] KV Cache is allocated. #tokens: 327310, K size: 2.50 GB, V size: 2.50 GB
[2026-04-14 16:08:49 TP0] Memory pool end. avail mem=4.49 GB
[2026-04-14 16:08:49 TP1] KV Cache is allocated. #tokens: 327310, K size: 2.50 GB, V size: 2.50 GB
[2026-04-14 16:08:49 TP1] Memory pool end. avail mem=4.49 GB
[2026-04-14 16:08:50 TP0] Current Python version 3.10 is below the recommended 3.11 version. It is recommended to upgrade to Python 3.11 or higher for the best experience.
[2026-04-14 16:08:50 TP0] Linear attention kernel backend: decode=triton, prefill=triton
[2026-04-14 16:08:50 TP0] Using hybrid linear attention backend for hybrid GDN models.
[2026-04-14 16:08:50 TP0] GDN kernel dispatcher: decode=TritonGDNKernel, extend=TritonGDNKernel, verify=TritonGDNKernel packed_decode=True
[2026-04-14 16:08:50 TP0] Capture cuda graph begin. This can take up to several minutes. avail mem=4.04 GB
[2026-04-14 16:08:50 TP0] Capture cuda graph bs [1, 2, 4, 8, 12, 16, 24]
[2026-04-14 16:08:50 TP1] Current Python version 3.10 is below the recommended 3.11 version. It is recommended to upgrade to Python 3.11 or higher for the best experience.
[2026-04-14 16:08:50 TP1] Using hybrid linear attention backend for hybrid GDN models.
[2026-04-14 16:08:50 TP1] Capture cuda graph begin. This can take up to several minutes. avail mem=4.04 GB

  0%|          | 0/7 [00:00<?, ?it/s]
Capturing batches (bs=24 avail_mem=4.00 GB):   0%|          | 0/7 [00:00<?, ?it/s]2026-04-14 16:08:51,862 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
[2026-04-14 16:08:51 TP0] Unexpected error during package walk: cutlass.cute.experimental
2026-04-14 16:08:51,940 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
[2026-04-14 16:08:51 TP1] Unexpected error during package walk: cutlass.cute.experimental

Capturing batches (bs=24 avail_mem=4.00 GB):  14%|█▍        | 1/7 [00:02<00:17,  2.96s/it]
Capturing batches (bs=16 avail_mem=3.93 GB):  14%|█▍        | 1/7 [00:02<00:17,  2.96s/it]
Capturing batches (bs=16 avail_mem=3.93 GB):  29%|██▊       | 2/7 [00:03<00:06,  1.37s/it]
Capturing batches (bs=12 avail_mem=3.89 GB):  29%|██▊       | 2/7 [00:03<00:06,  1.37s/it]
Capturing batches (bs=12 avail_mem=3.89 GB):  43%|████▎     | 3/7 [00:03<00:03,  1.24it/s]
Capturing batches (bs=8 avail_mem=3.88 GB):  43%|████▎     | 3/7 [00:03<00:03,  1.24it/s] 
Capturing batches (bs=8 avail_mem=3.88 GB):  57%|█████▋    | 4/7 [00:03<00:01,  1.84it/s]
Capturing batches (bs=4 avail_mem=3.85 GB):  57%|█████▋    | 4/7 [00:03<00:01,  1.84it/s]
Capturing batches (bs=4 avail_mem=3.85 GB):  71%|███████▏  | 5/7 [00:03<00:00,  2.50it/s]
Capturing batches (bs=2 avail_mem=3.84 GB):  71%|███████▏  | 5/7 [00:03<00:00,  2.50it/s]
Capturing batches (bs=2 avail_mem=3.84 GB):  86%|████████▌ | 6/7 [00:03<00:00,  3.08it/s]
Capturing batches (bs=1 avail_mem=3.80 GB):  86%|████████▌ | 6/7 [00:03<00:00,  3.08it/s]
Capturing batches (bs=1 avail_mem=3.80 GB): 100%|██████████| 7/7 [00:04<00:00,  3.31it/s]
Capturing batches (bs=1 avail_mem=3.80 GB): 100%|██████████| 7/7 [00:04<00:00,  1.72it/s]
[2026-04-14 16:08:55 TP1] Capture cuda graph end. Time elapsed: 4.78 s. mem usage=0.26 GB. avail mem=3.79 GB.
[2026-04-14 16:08:55 TP1] Disable piecewise CUDA graph because --disable-piecewise-cuda-graph is set
[2026-04-14 16:08:55 TP0] Capture cuda graph end. Time elapsed: 4.86 s. mem usage=0.26 GB. avail mem=3.79 GB.
[2026-04-14 16:08:55 TP0] Disable piecewise CUDA graph because --disable-piecewise-cuda-graph is set
[2026-04-14 16:09:08 TP0] max_total_num_tokens=327310, chunked_prefill_size=2048, max_prefill_tokens=16384, max_running_requests=62, context_len=262144, available_gpu_mem=3.79 GB
```

---

## Step 3: 记录启动后各卡显存

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free \
    --format=csv
```

**输出**：
```
index, memory.used [MiB], memory.free [MiB]
0, 20740 MiB, 3342 MiB
1, 8048 MiB, 14507 MiB
2, 20262 MiB, 3820 MiB
3, 8042 MiB, 16040 MiB
4, 18 MiB, 24063 MiB
```

**分析**：

### 理论值（来自 config.json 手算）


**config.json:**
```json
config.json 关键值：
  hidden_size (d)       = 4096
  num_hidden_layers (L) = 32
  num_attention_heads   = 16,  head_dim = 256
  num_key_value_heads   = 4    ← GQA，只有 4 组 KV
  intermediate_size     = 12288  ← FFN 隐层宽度
  vocab_size (V)        = 248320
```

#### 三大来源

1. Embedding 层

**V × d = 248320 × 4096 = 1.017B 参数**

把每个 token ID 映射成一个 d 维向量。词表越大参数越多。

2. 每个 Transformer 层（× 32）

Attention（4 个矩阵）：

```
Q:  d × (H × head_dim)    = 4096 × (16 × 256) = 16.8M
K:  d × (KV_H × head_dim) = 4096 × (4 × 256)  =  4.2M  ← GQA 少
V:  d × (KV_H × head_dim) = 4096 × (4 × 256)  =  4.2M  ← GQA 少
O:  (H × head_dim) × d    = (16×256) × 4096   = 16.8M
                                           合计 = 41.9M/层
```

FFN（SwiGLU，3 个矩阵）：

```
gate:  d × ffn = 4096 × 12288 = 50.3M
up:    d × ffn = 4096 × 12288 = 50.3M
down:  ffn × d = 12288 × 4096 = 50.3M
                              合计 = 151.0M/层
```

每层合计 ≈ 193M，× 32 层 = 6.17B

**合计**
| 组件 | 计算方式 | 参数量 |
|------|----------|--------|
| Embedding | 248320 × 4096 | 1.017B |
| Attn × 32 层 | (Q+K+V+O) × 32 = 41.9M × 32 | 1.342B |
| FFN × 32 层 | (gate+up+down) × 32 = 151M × 32 | 4.832B |
| **合计（共享LM Head）** | | **~7.2B** |

BF16 精度：7.2B × 2 bytes = **~14.4 GB 总权重**

**tp=2 时每卡权重**：14.4 GB ÷ 2 = **~7.2 GB/卡**

**KV cache pool**（`--mem-fraction-static 0.8`）：


### KV Cache Pool 精确计算（来自启动日志）

SGLang 源码核心公式（[model_runner_kv_cache_mixin.py](/root/wkspace/sglang/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py):65）：
```python
rest_memory = post_load_avail - pre_load_avail × (1 - mem_fraction_static)
```
含义：**加载权重后剩余内存** 减去 "**激活/workspace 预留**" = 可分配给 KV + Mamba 的内存。

> Mamba 详解: [06-mamba-ssm.md](../topics/06-mamba-ssm.md)。

代入日志数据：
```
pre_load_avail  = 22.93 GB   # 加载权重前可用（启动日志：Load weight begin）
post_load_avail = 14.08 GB   # 加载权重后可用（启动日志：Load weight end）
mem_fraction_static = 0.8    # 启动参数
mamba_full_memory_ratio = 0.9  # SGLang 默认值（server_args.py:561）

▶ 激活/workspace 预留 = 22.93 × (1 - 0.8) = 4.586 GB
▶ rest_memory = 14.08 - 4.59 = 9.49 GB   ← Mamba + KV 共享这些

```

#### 9.49 GB 如何在 Mamba 和 KV 之间分配

QQwen3.5-9B 是混合架构（Transformer 层 + Mamba/GDN 层），需要同时维护两种缓存。SGLang 用 `mamba_full_memory_ratio=0.9`（默认值）分配（源码 line 116-125）：
```
mamba_memory = rest × 0.9/(1+0.9) = 9.49 × 0.4737 = 4.497 GB ≈ 4.48 GB ✓
kv_memory    = rest × 1.0/(1+0.9) = 9.49 × 0.5263 = 4.997 GB ≈ 5.00 GB ✓
```

日志实测：
```
Mamba Cache: conv=0.10 GB + ssm=4.38 GB = 4.48 GB ✓
KV Cache:    K=2.50 GB + V=2.50 GB     = 5.00 GB ✓
```

#### 每 token KV 大小（反推架构）| 5.00 GB 能放多少 tokens？
```
per_token_kv = 5.00 GB × 1024³ / 327310 token = 16,018 bytes ≈ 16 KB/token
```

反推架构（需考虑 TP=2 的 KV head 分片）：
```
TP=2 时，每卡只存 kv_heads/tp = 4/2 = 2 个 KV head（per 层）

per_token_K_per_gpu = num_attn_layers × (kv_heads/tp) × head_dim × 2 bytes
                    = num_attn_layers × 2 × 256 × 2
                    = num_attn_layers × 1024 bytes

K size per GPU = 2.50 GB → per_token_K = 2.50 GB × 1024³ / 327310 = 8,011 bytes
8011 / 1024 = 7.82 ≈ 8 层
```

**结论：Qwen3.5-9B 32层中有 8 层是完整 Attention（每 4 层一个，由 config.json 的 full_attention_interval=4 确认），其余 24 层是 Mamba/GDN（用 SSM state）**。

> 之前误算为"4层"是因为漏掉了 TP=2 会把 kv_heads(=4) 分成每卡 2 个的效果。

#### 完整内存流水线
```
22.93 GB  启动前可用（CUDA context 约占 0.6 GB）
 - 8.85   权重（tp=2，每卡分到一半）
= 14.08   Load weight end
 - 4.586  预留给激活层（22.93 × 0.2）
= 9.49    rest_memory
 - 4.48   Mamba cache（186 并发请求的 SSM state）
 - 5.00   KV cache（327310 tokens × 16 KB/token）
= 4.60    Memory pool end ≈ 日志 4.49 GB ✓
 - 0.26   CUDA graphs
≈ 3.79   available_gpu_mem（日志最终值） ✓
```

### 实测 vs 理论对比

GPU 0/2 增量：20740 - 19 ≈ **+20.2 GB**

| 组件 | 大小 | 来源 |
|------|------|------|
| 权重（tp=2） | 8.85 GB | 日志 mem usage |
| Mamba cache | 4.48 GB | 日志 conv+ssm |
| KV pool | 5.00 GB | 日志 K+V |
| CUDA graphs | 0.26 GB | 日志 cuda graph |
| CUDA context + NCCL + 其他 | ~1.61 GB | 倒推 |
| **合计** | **~20.2 GB** | **≈ 实测 +20.2 GB ✓** |

---

## Step 4: 发送测试请求

```bash
# 基础测试
curl -s http://localhost:9991/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role": "user", "content": "用中文解释什么是 Tensor Parallelism"}],
        "max_tokens": 10000
    }' | jq . | tee -a answer.json

# 带 thinking 的请求（Qwen3 reasoning）
curl -s http://localhost:9991/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "max_tokens": 10000,
        "chat_template_kwargs": {"enable_thinking": true}
    }' | jq . | tee -a answer_thinking.json
```

**输出**：

- [answer.json](./answer.json)
- [answer_thinking.json](./answer_thinking.json)

### API 响应字段解析

```json
{
  "id": "c37110f...",          // 本次请求唯一 ID（调试/追踪用）
  "object": "chat.completion", // OpenAI 兼容类型标识，固定值
  "created": 1776153087,       // Unix 时间戳（秒）
  "model": "default",          // 模型名（未设置 --served-model-name 时为 "default"）
  "choices": [{
    "index": 0,                // 通常只有 1 个 choice（n=1）
    "message": {
      "role": "assistant",
      "content": "...",               // 实际回复文本
      "reasoning_content": "...",     // 【SGLang 扩展】thinking 过程（--reasoning-parser qwen3）
      "tool_calls": null              // function calling 未使用时为 null
    },
    "logprobs": null,          // 未开启 --return-logprob
    "finish_reason": "stop",   // 停止原因：stop=正常 | length=被 max_tokens 截断
    "matched_stop": 248046     // 【SGLang 扩展】触发停止的 token ID（Qwen3.5 的 <|im_end|>）
  }],
  "usage": {
    "prompt_tokens": 17,       // 输入 token 数
    "completion_tokens": 3280, // 输出 token 数（含 thinking）
    "total_tokens": 3297,
    "reasoning_tokens": 1626   // 【SGLang 扩展】其中 thinking 占用的 token 数
  },
  "metadata": {
    "weight_version": "default" // 【SGLang 扩展】权重标识
  }
}
```

### finish_reason 机制

模型不是被外部叫停的，而是**自己生成了停止 token**（类似于句号）：

```
正常 token → 正常 token → ... → <|im_end|>（ID 248046）
                                    ↑
                        SGLang 检测到 → 截断 → finish_reason: "stop"
```

- `matched_stop: 248046`：Qwen3.5-9B 词表大小 248320，特殊 token 分配在末尾区域，248046 即为 `<|im_end|>`（ChatML 轮次结束符）
- 使用 `--reasoning-parser qwen3` 时有两个关键 token：
  - `</think>`：结束 thinking 阶段，将之前内容写入 `reasoning_content`
  - `<|im_end|>`：结束整个生成

### max_tokens 上限

有两层限制：

| 限制层 | 来源 | 值 |
|--------|------|----|
| 模型架构上限 | config.json `max_position_embeddings` | 262144 tokens |
| KV cache pool（实际瓶颈） | 启动日志 `max_total_num_tokens` | 69316 tokens |

```
max_tokens 可填值上限  = 262144 - prompt_tokens ≈ 262127（API 不会拒绝）
单请求实际可输出上限   = 69316  - prompt_tokens ≈ 69299（超出则 KV pool 耗尽）

建议值：
  测试/学习  → max_tokens: 2000~4000
  长回复     → max_tokens: 10000~32000
  超长上下文 → 需减少并发或增大 --mem-fraction-static
```

---

## Step 5: 性能数据

**吞吐**：
- Prefill 速度：? tokens/s
- Decode 速度：? tokens/s

```bash
# 查看指标（如有开启 --enable-metrics）
curl http://localhost:9991/metrics 2>/dev/null | grep sglang
```

[metrics.log](metrics.log)

```bash
# 或查看日志中的吞吐数据
grep -i "throughput\|tokens/s\|tok/s"
```

**grep log 输出**：

[grep.log](grep.log)

---

## 结论

> 本节学习了：
> - 1. 如何启动 SGLang Serving 并 TP=2 加载 Qwen3.5-9B 模型；模型各部分显存占用计算方法：weights + KV cache + Mamba 预留
> - 2. API 请求和响应的结构和含义
> - 3. 模型生成过程中的一些细节（如 thinking、finish_reason、max_tokens 上限等）
> - 4. 性能指标的查看方法和初步数据
> - 5. Mamba 相关的日志输出和含义
---

## 下一步实验

- [ ] Exp-002：修改 `--tp-size 1`，对比单卡启动的显存和速度差异
- [ ] Exp-003：用 `--tp-size 2 --dp-size 2` 对比高并发下的吞吐
- [ ] Exp-004：发送长 context 请求（>10K tokens），观察 KV cache 动态增长
