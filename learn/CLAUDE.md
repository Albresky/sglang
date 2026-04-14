# SGLang 学习工作区

这个目录是 SGLang 系统性学习的知识库。每次开启新对话时，Claude 应自动加载此上下文。

## 环境概况

- 见 [environment.md](environment.md)：5× RTX 4090（24GB each，共 96GB），单机多卡
- SGLang 源码：`/root/wkspace/sglang/python/sglang/`
- 学习用模型：`Qwen/Qwen3.5-9B`（Qwen3 架构，9B 参数）
- SGLANG 官方 Claude Skills 文档：`/root/wkspace/sglang/python/sglang/.claude` `/root/wkspace/sglang/python/sglang/python/sglang/multimodal_gen/.claude`
- Python 用的 UV 虚拟环境 `/root/wkspace/sglang/.venv/`，已安装本 SGLang 源码及相关依赖
## 目录结构

```
learn/
├── CLAUDE.md            ← 本文件：学习规范与上下文（Claude 自动读取）
├── environment.md       ← GPU/CPU 环境详情
├── model.md             ← 模型启动命令参考
├── cheatsheet.md        ← 常用命令速查表
├── topics/              ← 理论学习笔记（每个主题一个文件）
│   ├── 01-tp-ep-pp-dp.md   并行策略
│   ├── 02-moe.md           MoE 架构
│   ├── 03-memory-calc.md   显存占用计算
│   ├── 04-transformer-arch.md  Transformer/LLM 架构
│   └── 05-logging-options.md   SGLang 日志与常用参数
└── experiments/         ← 动手实验记录（按编号命名）
    └── exp-001-qwen35-tp4-baseline/
        └── README.md     ← 实验目标、命令、输出分析、结论
```

## 学习规范（Claude 须遵守）

1. **理论学习** → 写入或更新对应 `topics/` 文件，结构为：概念解释 → SGLang 对应参数/代码 → 动手实验建议
2. **每次动手实验** → 创建新的 `experiments/exp-NNN-描述.md`，记录：目标、命令、关键输出、结论
3. **有新命令或技巧** → 更新 `cheatsheet.md`
4. **引用源码** → 优先查阅 `/root/wkspace/sglang/python/sglang/` 下的实际实现
5. **跨对话延续** → 每次对话开始时先检查 `experiments/` 最新记录，了解上次学到哪里

## 当前学习进度

- [ ] 完成 exp-001：Qwen3.5-9B 基线启动（TP=4）
- [ ] 理解 TP/EP/PP/DP 差异，能在本机设计合理的并行配置
- [ ] 能通过日志判断显存分配和 KV cache 使用情况
- [ ] 理解 MoE 路由机制及其与 EP 的关系
- [ ] 能手算一个模型的显存占用

## 快速上手

```bash
# 查看上次实验
ls learn/experiments/

# 启动 Qwen3.5-9B（TP=4）
python -m sglang.launch_server \
    --model-path 'Qwen/Qwen3.5-9B' \
    --port 9991 --tp-size 4 --mem-fraction-static 0.8 \
    --context-length 262144 --reasoning-parser qwen3

# 发送测试请求
curl http://localhost:9991/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"hello"}]}'
```
