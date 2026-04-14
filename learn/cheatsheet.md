# SGLang 速查表

## 启动命令

```bash
# Qwen3.5-9B，TP=4（推荐基线）
python -m sglang.launch_server \
    --model-path 'Qwen/Qwen3.5-9B' \
    --port 9991 --tp-size 4 --mem-fraction-static 0.8 \
    --context-length 262144 --reasoning-parser qwen3

# TP=1 单卡（对比实验）
python -m sglang.launch_server \
    --model-path 'Qwen/Qwen3.5-9B' \
    --port 9991 --tp-size 1

# TP=2 + DP=2（高并发场景）
python -m sglang.launch_server \
    --model-path 'Qwen/Qwen3.5-9B' \
    --port 9991 --tp-size 2 --dp-size 2
```

## 发送请求

```bash
# 基础对话
curl -s http://localhost:9991/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"hello"}],"max_tokens":100}'

# 流式输出
curl -s http://localhost:9991/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"hello"}],"stream":true}'

# 查询模型信息
curl http://localhost:9991/get_model_info

# 健康检查
curl http://localhost:9991/health
```

## 显存监控

```bash
# 实时监控
watch -n 2 nvidia-smi

# 单次查询（CSV 格式）
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total \
    --format=csv,noheader,nounits

# SGLang 指标（需 --enable-metrics）
curl http://localhost:9991/metrics | grep sglang
```

## 日志过滤

```bash
# 过滤显存相关
grep -i "memory\|cache\|alloc" /tmp/sglang.log

# 过滤吞吐相关
grep -i "throughput\|tok/s\|tokens/s" /tmp/sglang.log

# 过滤 TP 初始化
grep -i "rank\|nccl\|tensor_parallel" /tmp/sglang.log

# 过滤错误
grep -i "error\|exception\|failed" /tmp/sglang.log
```

## 模型信息

```bash
# 查看 config.json
cat ~/.cache/huggingface/hub/models--HauhauCS--Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/snapshots/*/config.json 2>/dev/null | python -m json.tool

# 查看参数量
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('Qwen/Qwen3.5-9B')
print(cfg)
"
```

## 显存估算

```python
# 快速估算（BF16）
params_b = 9      # 参数量（B）
tp = 4            # TP 卡数
weight_per_gpu = params_b * 2 / tp  # GB
print(f"每卡权重显存: {weight_per_gpu:.1f} GB")
print(f"每卡剩余（KV cache）: {24 - weight_per_gpu:.1f} GB")
```

## 常见问题

| 问题 | 解决方法 |
|------|----------|
| OOM（显存不足） | 减小 `--context-length` 或 `--mem-fraction-static` |
| 启动慢 | 首次加载正常，后续有缓存 |
| TP 不整除报错 | num_attention_heads 必须能被 tp-size 整除 |
| 端口占用 | `lsof -i :9991` 查找并 kill |

## 学习笔记索引

| 主题 | 文件 |
|------|------|
| TP/EP/PP/DP 并行 | [topics/01-tp-ep-pp-dp.md](topics/01-tp-ep-pp-dp.md) |
| MoE 架构 | [topics/02-moe.md](topics/02-moe.md) |
| 显存计算 | [topics/03-memory-calc.md](topics/03-memory-calc.md) |
| Transformer 架构 | [topics/04-transformer-arch.md](topics/04-transformer-arch.md) |
| 日志与参数 | [topics/05-logging-options.md](topics/05-logging-options.md) |
| Mamba / SSM 架构 | [topics/06-mamba-ssm.md](topics/06-mamba-ssm.md) |
| 基线实验 | [experiments/exp-001-qwen35-tp2-baseline/README.md](experiments/exp-001-qwen35-tp2-baseline/README.md) |
