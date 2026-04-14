# 显存占用计算方法

## 三大组成部分

```
总显存 = 模型权重 + KV Cache + 激活值（activation）
```

推理时激活值很小（相比训练），主要关注前两项。

---

## 1. 模型权重显存

### 公式
```
权重显存 (GB) = 参数量 × 每参数字节数 / 1e9
```

### 数据类型对照
| 数据类型 | 字节数 | 说明 |
|----------|--------|------|
| float32 (FP32) | 4 | 训练常用 |
| bfloat16 (BF16) | 2 | 推理主流 |
| float16 (FP16) | 2 | 推理主流 |
| int8 | 1 | 量化 |
| int4 / NF4 | 0.5 | 激进量化 |

### 示例：Qwen3.5-9B（BF16）
```
9B 参数 × 2 字节 = 18 GB
```

实际会略大（embedding、LayerNorm 等），通常按 **参数量 × 2.1** 估算。

### 实际验证命令
```bash
# 启动后查看每卡显存
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

# 或用 watch 实时监控
watch -n 1 nvidia-smi
```

---

## 2. KV Cache 显存

### 原理
Transformer 推理时，每个 Token 的 Key 和 Value 会被缓存，避免重复计算。

### 公式
```
KV Cache (bytes) =
  2                    # K + V
  × num_layers
  × num_kv_heads       # GQA 模型用 num_kv_heads（< num_heads）
  × head_dim
  × max_seq_len
  × batch_size
  × 每元素字节数（通常 2，即 BF16）
```

### Qwen3.5-9B 参数（需查 config.json）
```python
# 典型值（待从 model config 读取后填入）
num_layers = 36
num_kv_heads = 8          # GQA，比 num_heads=16 少
head_dim = 128
```

### 示例计算
```python
num_layers   = 36
num_kv_heads = 8
head_dim     = 128
max_seq_len  = 8192
batch_size   = 1
bytes_per_elem = 2  # BF16

kv_cache = (2 * num_layers * num_kv_heads * head_dim * max_seq_len * batch_size * bytes_per_elem)
kv_cache_gb = kv_cache / 1e9
# 约 0.6 GB（单条 8K seq）

# context_length=262144 时每条请求：
# 2 × 36 × 8 × 128 × 262144 × 2 / 1e9 ≈ 19.3 GB ！
# 所以长上下文需要非常多显存
```

### SGLang 控制参数
```bash
--mem-fraction-static 0.8   # 80% 显存给 KV cache pool（剩余 20% 留给权重+激活）
--context-length 262144      # 最大 context 长度
--max-running-requests 10    # 最大并发请求数
```

---

## 3. SGLang 显存分配策略

SGLang 使用**静态预分配**策略：
1. 启动时加载模型权重（固定）
2. 剩余显存 × `mem-fraction-static` 全部用于 KV cache pool
3. 运行时按需从 pool 中分配 KV cache block

### 查看 SGLang 的显存分配日志
启动时会打印：
```
[INFO] Memory pool: XX GB allocated for KV cache
[INFO] Max tokens in cache: XXXXXX
```

---

## 4. TP 对显存的影响

TP=4 时，每张卡只存 1/4 的权重：
```
单卡权重显存 = 总权重显存 / tp-size
Qwen3.5-9B BF16, tp=4: 18GB / 4 = 4.5 GB/卡（用于权重）
剩余: 24 - 4.5 = 19.5 GB 可用于 KV cache
```

---

## 5. 快速估算脚本

```python
def estimate_memory(
    params_b,        # 参数量（单位：B，十亿）
    dtype_bytes=2,   # BF16=2, INT8=1, INT4=0.5
    tp_size=1,
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
    max_seq_len=8192,
    batch_size=1,
):
    weight_gb = params_b * 1e9 * dtype_bytes / 1e9 / tp_size
    kv_gb = (2 * num_layers * num_kv_heads * head_dim *
             max_seq_len * batch_size * dtype_bytes) / 1e9
    print(f"权重显存（每卡）: {weight_gb:.1f} GB")
    print(f"KV Cache: {kv_gb:.2f} GB（单请求 {max_seq_len} tokens）")
    print(f"合计估算: {weight_gb + kv_gb:.1f} GB")

# Qwen3.5-9B, tp=4
estimate_memory(9, tp_size=4, num_layers=36, num_kv_heads=8, max_seq_len=8192)
```

---

## 动手实验

启动服务器后运行：
```bash
# 记录启动后各卡基线显存
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# 发送长请求，观察 KV cache 增长
# （见 experiments/ 记录）
```

## 结论/心得

> 待填写
