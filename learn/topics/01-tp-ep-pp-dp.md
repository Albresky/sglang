# 并行策略：TP / EP / PP / DP

## 概念速览

| 缩写 | 全称 | 切分维度 | 典型场景 |
|------|------|----------|----------|
| TP | Tensor Parallelism | 单层权重矩阵按列/行切分 | 单节点多卡，减少单卡显存 |
| EP | Expert Parallelism | MoE 的 Expert 分布到不同卡 | MoE 模型，Expert 数量多 |
| PP | Pipeline Parallelism | 模型层按阶段切分 | 超大模型跨节点，层数多 |
| DP | Data Parallelism | 同一模型复制多份，处理不同请求 | 高并发推理，吞吐优先 |

---

## Tensor Parallelism (TP)

### 原理
将 Transformer 中的线性层（如 `QKV projection`、`FFN`）按列或行切分到多张 GPU。
- **列切分**（Column Parallel）：`W` 切成 `[W1 | W2 | ... | Wn]`，每卡各算一部分，最后 `AllGather`
- **行切分**（Row Parallel）：输入切分，每卡各算一部分，最后 `AllReduce`

### 通信开销
每个 Transformer 层的前向/后向各需要 **2次 AllReduce**（Megatron-LM 方案）。
TP=8 通信量远大于 TP=4，建议 TP 不超过单机 GPU 数（NVLink 带宽 > PCIe）。

### SGLang 参数
```bash
--tp-size 4   # 使用 4 卡做 Tensor Parallel
```

### 本机配置建议（5× RTX 4090）
```bash
# Qwen3.5-9B（~18GB FP16）4卡可装下，留 1 卡备用
--tp-size 4

# 如果想跑更大模型（如 70B）可尝试
--tp-size 5   # 但 5 不是 2 的幂次，部分 kernel 效率略低
```

### 源码位置
- `python/sglang/srt/model_executor/model_runner.py` — TP 初始化
- `python/sglang/srt/layers/linear.py` — ColumnParallelLinear / RowParallelLinear

---

## Expert Parallelism (EP)

### 原理
用于 **MoE（Mixture of Experts）模型**。模型有 N 个 Expert，将这 N 个 Expert 均匀分配到 E 张 GPU。
- 每张 GPU 只存储 N/E 个 Expert 的权重
- Token 经过 Router 后，发送到对应 Expert 所在的 GPU（`All2All` 通信）

### TP vs EP 的关系
- TP 切的是**每个 Expert 内部**的矩阵
- EP 切的是 **Expert 之间**的分配
- 二者可以组合：`tp=2, ep=4` 表示 4 个 Expert 组，每组用 2 卡做 TP

### SGLang 参数
```bash
--ep-size 4   # 4 卡做 Expert Parallel（需要 MoE 模型）
```

### 动手实验建议
> Qwen3.5-9B 不是 MoE 模型，EP 实验需要换 MoE 模型（如 Qwen2-57B-A14B 或 DeepSeek-V2-Lite）

---

## Pipeline Parallelism (PP)

### 原理
将模型的层（Layer）切成若干段（Stage），每段放一张/组 GPU。
- GPU 0 跑 Layer 0~15，GPU 1 跑 Layer 16~31，...
- 前一段的输出（activation）通过 `P2P 通信` 传给下一段

### 优缺点
- 优点：每张 GPU 只需存一部分层的权重
- 缺点：**bubble time**（流水线空泡），低 batch size 时 GPU 利用率低

### SGLang 参数
```bash
--pp-size 2   # 2 阶段流水线
```

### 注意
PP 通常用于**训练**或**超大模型跨节点推理**，单机推理场景 TP 更常用。

---

## Data Parallelism (DP)

### 原理
启动多个完整模型副本，每个副本处理不同的请求。负载均衡器（如 SGLang 的 `dp_worker`）将请求分发。

### SGLang 参数
```bash
--dp-size 2   # 2 个 DP worker，每个各跑一套完整模型
# 总 GPU = tp-size × dp-size
```

### 适用场景
- 模型较小（单卡或少量卡可装下）
- 需要高吞吐（大量并发请求）

---

## 本机实验矩阵

| 配置 | GPU 用量 | 适用模型大小 | 说明 |
|------|----------|--------------|------|
| `--tp-size 1` | 1 卡 | ≤ 20GB | 测试用 |
| `--tp-size 4` | 4 卡 | ≤ 80GB | Qwen3.5-9B 推荐配置 |
| `--tp-size 2 --dp-size 2` | 4 卡 | ≤ 40GB | 高并发场景 |
| `--tp-size 5` | 5 卡 | ≤ 100GB | 极限配置 |

---

## 动手实验

见 [../experiments/exp-001-qwen35-tp2-baseline/README.md](../experiments/exp-001-qwen35-tp2-baseline/README.md)

## 结论/心得

> 待填写：完成实验后记录观察到的现象、踩过的坑
