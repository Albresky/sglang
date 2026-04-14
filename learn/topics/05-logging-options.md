# SGLang 日志解读与常用参数

## 启动日志关键段落

### 1. 模型加载日志
```
[INFO] Loading model from ...
[INFO] Model loaded in X.Xs
```

### 2. 显存分配日志
```
[INFO] Memory usage: X.X GB (model) + X.X GB (KV cache pool)
[INFO] KV cache: X blocks, each block = X tokens
[INFO] Max total tokens in cache: XXXXXX
```
**含义**：可以推算出最大并发能处理多少 token

### 3. TP 初始化日志
```
[INFO] Tensor parallel size: 4
[INFO] Initializing process group with backend nccl
[INFO] Rank 0/1/2/3 initialized
```

### 4. 请求处理日志（开启 verbose 后）
```
[INFO] Prefill: X tokens, latency: X ms
[INFO] Decode: X tokens generated, throughput: X tok/s
```

---

## 常用启动参数速查

### 基础参数
```bash
--model-path PATH          # 模型路径或 HuggingFace model ID
--port PORT                # 服务端口（默认 30000）
--host HOST                # 绑定地址（默认 127.0.0.1，局域网用 0.0.0.0）
```

### 并行参数
```bash
--tp-size N                # Tensor Parallel 卡数
--dp-size N                # Data Parallel 副本数
--pp-size N                # Pipeline Parallel 阶段数
--ep-size N                # Expert Parallel（仅 MoE）
```

### 显存参数
```bash
--mem-fraction-static F    # KV cache 占用显存比例（0.0~1.0，默认 0.9）
--context-length N         # 最大上下文长度
--max-running-requests N   # 最大并发请求数
--max-total-tokens N       # KV cache 池总 token 容量（覆盖 mem-fraction-static）
```

### 推理精度
```bash
--dtype auto               # 自动选择（通常 BF16）
--dtype float16            # 强制 FP16
--quantization awq         # AWQ 量化
--quantization gptq        # GPTQ 量化
--quantization fp8         # FP8 量化（需要 H100/H200）
```

### 调试与日志
```bash
--log-level info           # 日志级别（debug/info/warning/error）
--log-level-http warning   # HTTP 层日志级别（减少请求日志刷屏）
--show-time-cost           # 显示各阶段耗时
--enable-metrics           # 开启 Prometheus 指标（默认端口 +1）
```

### 功能特性
```bash
--reasoning-parser qwen3   # 启用 reasoning 解析（Qwen3 思考模式）
--disable-flashinfer       # 禁用 FlashInfer（调试用）
--enable-torch-compile     # 开启 torch.compile 加速（首次慢）
--chunked-prefill-size N   # Chunked prefill 块大小
```

---

## 运行时监控

### 实时显存
```bash
watch -n 2 nvidia-smi
```

### SGLang 指标接口（开启 --enable-metrics 后）
```bash
curl http://localhost:9991/metrics   # Prometheus 格式

# 关键指标：
# sglang:num_running_reqs        当前运行中的请求数
# sglang:token_usage             KV cache 使用率
# sglang:gen_throughput          生成吞吐（tok/s）
# sglang:ttft_seconds            Time To First Token
```

### 健康检查
```bash
curl http://localhost:9991/health
curl http://localhost:9991/get_model_info
```

---

## 日志分析技巧

### 找显存瓶颈
```bash
# 启动时加 --log-level debug，过滤显存相关
python -m sglang.launch_server ... 2>&1 | grep -i "memory\|cache\|alloc"
```

### 查看请求调度
```bash
python -m sglang.launch_server ... 2>&1 | grep -i "prefill\|decode\|batch"
```

### 查看 TP 通信
```bash
python -m sglang.launch_server ... 2>&1 | grep -i "rank\|nccl\|barrier"
```

---

## 结论/心得

> 待填写：记录实际遇到的有用日志片段
