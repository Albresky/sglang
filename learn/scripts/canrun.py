#!/usr/bin/env python3
"""
canrun.py - 判断 HF 模型是否能在当前服务器上运行

用法:
  python canrun.py                      # 列出已下载模型，交互选择
  python canrun.py Qwen/Qwen3.5-9B      # 直接指定模型 ID
  python canrun.py Qwen/Qwen3.5-9B --dtype int4 --tp 2
  python canrun.py --list               # 仅列出已下载模型
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import unicodedata

# ── 终端宽度感知的对齐工具 ────────────────────────────────────────────────────

def _dw(s: str) -> int:
    """字符串在终端的实际显示宽度（CJK 宽字符算 2 列）。"""
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1
               for c in str(s))

def _rj(s, w: int) -> str:
    """右对齐到显示宽度 w（用空格填充）。"""
    s = str(s)
    return ' ' * max(0, w - _dw(s)) + s

def _lj(s, w: int) -> str:
    """左对齐到显示宽度 w（用空格填充）。"""
    s = str(s)
    return s + ' ' * max(0, w - _dw(s))

def _sep(w: int) -> str:
    """生成显示宽度为 w 的横线分隔符（ASCII 连字符）。"""
    return '-' * w

# ── 硬件检测 ──────────────────────────────────────────────────────────────────

def detect_gpus():
    """返回每张 GPU 的显存信息，包含总量、已用、剩余（GB）。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        gpus = []
        for line in out.strip().splitlines():
            idx, name, total_mb, used_mb, free_mb = [x.strip() for x in line.split(",", 4)]
            gpus.append({
                "index":    int(idx),
                "name":     name,
                "vram_gb":  int(total_mb) / 1024,   # 总量
                "used_gb":  int(used_mb)  / 1024,   # 已用
                "free_gb":  int(free_mb)  / 1024,   # 剩余
            })
        return gpus
    except Exception:
        return []


def detect_cpu_ram_gb():
    """从 /proc/meminfo 读取 CPU 总内存（GB）。"""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / 1024 / 1024
    except Exception:
        pass
    return None


# ── HF 缓存扫描 ───────────────────────────────────────────────────────────────

def get_hf_cache_dir():
    return os.environ.get("HF_HOME",
           os.environ.get("HUGGINGFACE_HUB_CACHE",
           os.path.expanduser("~/.cache/huggingface/hub")))


def list_cached_models():
    """扫描 HF cache，返回 (model_id, snapshot_dir, config_path) 列表。"""
    cache_dir = get_hf_cache_dir()
    results = []
    for entry in sorted(os.listdir(cache_dir)):
        if not entry.startswith("models--"):
            continue
        # models--Org--Name → Org/Name
        parts = entry[len("models--"):].split("--", 1)
        model_id = "/".join(parts) if len(parts) == 2 else parts[0]

        # 找最新 snapshot
        snap_base = os.path.join(cache_dir, entry, "snapshots")
        if not os.path.isdir(snap_base):
            continue
        snaps = sorted(os.listdir(snap_base))
        if not snaps:
            continue
        snap_dir = os.path.join(snap_base, snaps[-1])
        config_path = os.path.join(snap_dir, "config.json")

        results.append((model_id, snap_dir, config_path if os.path.exists(config_path) else None))
    return results


# ── config.json 解析 ──────────────────────────────────────────────────────────

def load_config(config_path):
    """加载并拍平 config.json（处理 text_config 嵌套）。"""
    with open(config_path) as f:
        cfg = json.load(f)
    # 部分多模态/MoE 模型把语言模型参数放在 text_config 里
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        merged = {**cfg["text_config"], **{k: v for k, v in cfg.items() if k != "text_config"}}
        return merged
    return cfg


def parse_architecture(cfg):
    """从 config 提取关键架构参数，返回 dict。"""
    hidden      = cfg.get("hidden_size", 0)
    num_layers  = cfg.get("num_hidden_layers", 0)
    num_heads   = cfg.get("num_attention_heads", 1)
    kv_heads    = cfg.get("num_key_value_heads", num_heads)
    vocab       = cfg.get("vocab_size", 0)
    head_dim    = cfg.get("head_dim", hidden // num_heads if num_heads else 0)
    intermediate = cfg.get("intermediate_size", 0)
    model_type  = cfg.get("model_type", "unknown")
    max_pos     = cfg.get("max_position_embeddings", 0)

    # Full attention 层数（混合架构）
    layer_types = cfg.get("layer_types", [])
    if layer_types:
        num_full_attn = sum(1 for t in layer_types if t == "full_attention")
    elif "full_attention_interval" in cfg:
        interval = cfg["full_attention_interval"]
        num_full_attn = num_layers // interval
    else:
        num_full_attn = num_layers  # 纯 Transformer

    is_hybrid = num_full_attn < num_layers

    # MoE 参数
    is_moe = "num_experts" in cfg
    num_experts         = cfg.get("num_experts", 0)
    num_experts_per_tok = cfg.get("num_experts_per_tok", 0)
    moe_intermediate    = cfg.get("moe_intermediate_size", intermediate)
    shared_intermediate = cfg.get("shared_expert_intermediate_size", 0)

    return {
        "model_type":       model_type,
        "hidden":           hidden,
        "num_layers":       num_layers,
        "num_heads":        num_heads,
        "kv_heads":         kv_heads,
        "head_dim":         head_dim,
        "intermediate":     intermediate,
        "vocab":            vocab,
        "max_pos":          max_pos,
        "num_full_attn":    num_full_attn,
        "is_hybrid":        is_hybrid,
        "is_moe":           is_moe,
        "num_experts":      num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "moe_intermediate": moe_intermediate,
        "shared_intermediate": shared_intermediate,
    }


def estimate_params_b(arch):
    """从架构参数估算总参数量（B）。"""
    h = arch["hidden"]
    layers = arch["num_layers"]
    kv = arch["kv_heads"]
    heads = arch["num_heads"]
    d = arch["head_dim"]
    inter = arch["intermediate"]
    vocab = arch["vocab"]

    # 注意力参数/层（Q + K + V + O）
    attn = h * heads * d + h * kv * d * 2 + h * h

    if arch["is_moe"]:
        n_exp = arch["num_experts"]
        moe_i = arch["moe_intermediate"]
        shared_i = arch["shared_intermediate"]
        ffn = n_exp * h * moe_i * 3 + (h * shared_i * 3 if shared_i else 0)
    else:
        ffn = h * inter * 3  # SwiGLU

    # 对混合模型：线性注意力层的参数简单用 ffn 估算（忽略 SSM state dim 误差）
    layer_params = (attn + ffn) * layers
    emb = vocab * h * 2
    return (layer_params + emb) / 1e9


def get_weight_size_gb(snap_dir):
    """累加 safetensors/bin 文件大小（= 实际存储的权重字节数）。"""
    total = sum(os.path.getsize(p) for p in glob.glob(os.path.join(snap_dir, "*.safetensors")))
    if total == 0:
        total = sum(os.path.getsize(p) for p in glob.glob(os.path.join(snap_dir, "*.bin"))
                    if not p.endswith("index.bin"))
    return total / 1024**3


# ── 核心分析 ──────────────────────────────────────────────────────────────────

DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}


def analyze(model_id, snap_dir, config_path, dtype_override, gpu_vram_list, cpu_ram_gb):
    """gpu_vram_list: list of dicts with keys index/name/vram_gb; optionally used_gb/free_gb."""
    print()
    print("─" * 60)
    print(f"  模型: {model_id}")
    print("─" * 60)

    # ── 架构信息 ──
    if config_path:
        cfg  = load_config(config_path)
        arch = parse_architecture(cfg)
        calc_params = estimate_params_b(arch)
    else:
        cfg  = {}
        arch = {}
        calc_params = None
        print("  [!] 未找到 config.json，无法解析架构")

    disk_gb = get_weight_size_gb(snap_dir)
    file_dtype = cfg.get("dtype", cfg.get("torch_dtype", "bf16")).lower().replace("float", "f").replace("bfloat", "bf")

    print()
    print("【架构】")
    if arch:
        hybrid_str = f"混合 {arch['num_full_attn']}/{arch['num_layers']} full-attn" if arch["is_hybrid"] else "纯 Transformer"
        moe_str    = f"  MoE {arch['num_experts']}E top{arch['num_experts_per_tok']}" if arch["is_moe"] else ""
        print(f"  model_type:   {arch['model_type']}{moe_str}")
        print(f"  layers:       {arch['num_layers']}  ({hybrid_str})")
        print(f"  hidden:       {arch['hidden']}  heads: {arch['num_heads']}  kv_heads: {arch['kv_heads']}  head_dim: {arch['head_dim']}")
        if not arch["is_moe"]:
            print(f"  intermediate: {arch['intermediate']}")
        else:
            print(f"  moe_intermediate: {arch['moe_intermediate']}  experts: {arch['num_experts']}  shared: {arch['shared_intermediate']}")
        print(f"  vocab:        {arch['vocab']}  max_ctx: {arch['max_pos']//1024}K")
        if calc_params:
            print(f"  参数量(估算): ~{calc_params:.1f}B")

    # ── 磁盘文件 → 权重精度 ──
    print()
    print("【权重文件】")
    if disk_gb > 0:
        inferred_dtype = file_dtype if file_dtype in DTYPE_BYTES else "bf16"
        bparam_disk    = DTYPE_BYTES.get(inferred_dtype, 2)
        params_from_disk = disk_gb / bparam_disk if bparam_disk else None
        print(f"  磁盘大小:  {disk_gb:.1f} GB  (推断精度: {inferred_dtype})")
        if params_from_disk:
            print(f"  实际参数量(反推): ~{params_from_disk:.1f}B")
        stored_dtype = inferred_dtype
        stored_gb    = disk_gb
    else:
        print("  [!] 未找到权重文件（可能仅有 config 或 GGUF 格式）")
        stored_dtype = "bf16"
        stored_gb    = (calc_params * 2) if calc_params else 0

    # ── 推理精度 ──
    infer_dtype = dtype_override or stored_dtype
    if infer_dtype not in DTYPE_BYTES:
        infer_dtype = "bf16"
    bparam = DTYPE_BYTES[infer_dtype]

    # 若推理精度 ≠ 存储精度，重新算权重大小
    if infer_dtype != stored_dtype and calc_params:
        weight_total_gb = calc_params * bparam
    elif stored_gb > 0:
        bparam_stored = DTYPE_BYTES.get(stored_dtype, 2)
        weight_total_gb = stored_gb * (bparam / bparam_stored)
    else:
        weight_total_gb = 0

    if dtype_override and dtype_override != stored_dtype:
        print(f"  推理精度:  {infer_dtype}  (覆盖存储精度 {stored_dtype}, 权重 ~{weight_total_gb:.1f} GB)")

    # ── 硬件状态表 ──
    num_gpus  = len(gpu_vram_list)
    gpu_names = ", ".join(set(g["name"] for g in gpu_vram_list)) if gpu_vram_list else "未知"
    has_free  = "free_gb" in (gpu_vram_list[0] if gpu_vram_list else {})

    print()
    print("【硬件】")
    if cpu_ram_gb:
        cpu_ok = weight_total_gb * 1.2 <= cpu_ram_gb
        print(f"  CPU RAM: {cpu_ram_gb:.0f} GB  (加载需 ~{weight_total_gb*1.2:.0f} GB  {'✓' if cpu_ok else '✗ 不足！'})")
    print()

    # 列宽（单位：终端显示列数）
    # 硬件表
    CW_IDX, CW_NAME, CW_MEM = 4, 28, 6
    if has_free:
        hdr = (f"  {_rj('GPU', CW_IDX)}  {_lj('型号', CW_NAME)}"
               f"  {_rj('总量', CW_MEM)}  {_rj('已用', CW_MEM)}  {_rj('剩余', CW_MEM)}  状态")
        sep = (f"  {_sep(CW_IDX)}  {_sep(CW_NAME)}"
               f"  {_sep(CW_MEM)}  {_sep(CW_MEM)}  {_sep(CW_MEM)}")
        print(hdr)
        print(sep)
        for g in gpu_vram_list:
            busy = g["used_gb"] > 0.5
            flag = "⚠ 占用" if busy else "✓"
            print(f"  {g['index']:>{CW_IDX}}  {_lj(g['name'], CW_NAME)}"
                  f"  {g['vram_gb']:>{CW_MEM-1}.0f}G"
                  f"  {g['used_gb']:>{CW_MEM-1}.1f}G"
                  f"  {g['free_gb']:>{CW_MEM-1}.1f}G  {flag}")
        sorted_gpus = sorted(gpu_vram_list, key=lambda g: g["free_gb"], reverse=True)
        print()
        print("  按剩余显存排序: " +
              ", ".join(f"GPU{g['index']}({g['free_gb']:.1f}G)" for g in sorted_gpus))
    else:
        gpu_vram_gb = gpu_vram_list[0]["vram_gb"] if gpu_vram_list else 24.0
        print(f"  {num_gpus}× {gpu_names}  {gpu_vram_gb:.0f} GB each")

    # ── per-token KV 大小（全量，不含 TP 分片）──
    if arch:
        n_attn  = arch["num_full_attn"]
        kv_h    = arch["kv_heads"]
        hd      = arch["head_dim"]
        per_tok = n_attn * 2 * kv_h * hd * bparam  # bytes (全量 KV，TP=1 基准)
    else:
        per_tok = 0

    is_hybrid_moe = arch.get("is_hybrid") or arch.get("is_moe")

    # TP 分析表列宽（终端显示列数）
    CW_TP, CW_GPU, CW_PRE, CW_W, CW_KV, CW_TOK, CW_ST = 4, 13, 9, 8, 8, 10, 3

    def tp_analysis(label, get_avail_gb, tp_range):
        """
        打印一个 TP 分析表。
        get_avail_gb(tp) → (avail_gb_per_gpu, gpu_desc_str, imbalance_warning)
        """
        print()
        print(f"【{label}】  (权重 {weight_total_gb:.1f} GB {infer_dtype})")
        hdr = (f"  {_rj('TP', CW_TP)}  {_rj('选用GPU', CW_GPU)}"
               f"  {_rj('预装前/卡', CW_PRE)}  {_rj('权重/卡', CW_W)}"
               f"  {_rj('KV pool', CW_KV)}  {_rj('最大tokens', CW_TOK)}"
               f"  {_rj('状', CW_ST)}  备注")
        sep = (f"  {_sep(CW_TP)}  {_sep(CW_GPU)}"
               f"  {_sep(CW_PRE)}  {_sep(CW_W)}"
               f"  {_sep(CW_KV)}  {_sep(CW_TOK)}"
               f"  {_sep(CW_ST)}")
        print(hdr)
        print(sep)

        best_tp = None
        for tp in tp_range:
            avail_gb, gpu_desc, warn = get_avail_gb(tp)

            w_per_gpu = weight_total_gb / tp
            post_load  = avail_gb - w_per_gpu
            act_res    = avail_gb * (1 - 0.8)
            rest       = post_load - act_res      # avail×0.8 - weight

            kv_pool    = rest / (1 + 0.9) if is_hybrid_moe else rest

            per_tok_gpu = per_tok / tp if per_tok > 0 else 0
            max_tok_k   = (kv_pool * 1024**3 / per_tok_gpu / 1000
                           if per_tok_gpu > 0 and kv_pool > 0 else 0)

            vram_ok = w_per_gpu < avail_gb and kv_pool > 0
            status  = "✓" if vram_ok else "✗"
            if warn:
                status = "⚠"

            note = warn or ""
            if vram_ok and not warn:
                if best_tp is None:
                    best_tp = tp
                    note = "← 最小 TP"
                elif tp == tp_range[-1]:
                    note = "全卡"

            tok_str  = f"~{max_tok_k:.0f}K" if max_tok_k > 0 else "N/A"
            avail_s  = f"{avail_gb:.1f}G"
            w_s      = f"{w_per_gpu:.1f}G"
            kv_s     = f"{kv_pool:.1f}G"
            print(f"  {tp:>{CW_TP}}  {_rj(gpu_desc, CW_GPU)}"
                  f"  {_rj(avail_s, CW_PRE)}  {_rj(w_s, CW_W)}"
                  f"  {_rj(kv_s, CW_KV)}  {_rj(tok_str, CW_TOK)}"
                  f"  {_rj(status, CW_ST)}  {note}")

        return best_tp

    # ── 理论分析（空卡，按总量计算）──
    def avail_theoretical(tp):
        g = gpu_vram_list[0] if gpu_vram_list else {"vram_gb": 24.0}
        # SGLang 启动时 CUDA context 约占 0.5-1 GB，用 total×0.97 近似
        avail = g["vram_gb"] * 0.97
        return avail, f"任意{tp}卡", ""

    best_tp_theory = tp_analysis(
        "理论分析（空卡，按总量估算）",
        avail_theoretical,
        list(range(1, num_gpus + 1)),
    )

    # ── 当前分析（按实际剩余显存）──
    if has_free:
        sorted_by_free = sorted(gpu_vram_list, key=lambda g: g["free_gb"], reverse=True)

        def avail_current(tp):
            # 选剩余最多的 tp 张卡
            chosen  = sorted_by_free[:tp]
            min_free = min(g["free_gb"] for g in chosen)
            max_free = max(g["free_gb"] for g in chosen)
            gpu_ids  = ",".join(str(g["index"]) for g in chosen)

            # 内存不均衡警告（SGLang 会拒绝差异过大的配置）
            warn = ""
            if max_free > 0 and min_free / max_free < 0.7:
                warn = f"不均衡!"
            return min_free, gpu_ids, warn

        best_tp_current = tp_analysis(
            "当前分析（实际剩余 VRAM，选最空闲卡）",
            avail_current,
            list(range(1, num_gpus + 1)),
        )
    else:
        best_tp_current = best_tp_theory

    # ── 推荐命令 ──
    best_tp = best_tp_current or best_tp_theory
    print()
    if best_tp is None:
        print("  ✗ 结论: 无法运行，考虑量化（int4/fp8）或释放 GPU 显存")
    else:
        # 如果当前模式，给出 CUDA_VISIBLE_DEVICES 建议
        cuda_prefix = ""
        if has_free:
            chosen = sorted(gpu_vram_list, key=lambda g: g["free_gb"], reverse=True)[:best_tp]
            ids = ",".join(str(g["index"]) for g in chosen)
            cuda_prefix = f"CUDA_VISIBLE_DEVICES={ids} "

        print(f"  ✓ 推荐 TP={best_tp}，启动命令：")
        print()
        print(f"    {cuda_prefix}HF_HUB_OFFLINE=1 python -m sglang.launch_server \\")
        print(f"        --model-path '{model_id}' \\")
        print(f"        --port 9991 --tp-size {best_tp} --mem-fraction-static 0.8")
    print()


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="判断 HF 模型能否在本机运行")
    parser.add_argument("model", nargs="?", help="模型 ID，如 Qwen/Qwen3.5-9B")
    parser.add_argument("--dtype", default=None,
                        choices=list(DTYPE_BYTES.keys()),
                        help="推理精度（覆盖模型默认精度）")
    parser.add_argument("--tp", type=int, default=None,
                        help="只分析指定 TP 数（默认分析所有）")
    parser.add_argument("--list", action="store_true", help="仅列出已下载模型")
    args = parser.parse_args()

    gpus       = detect_gpus()
    cpu_ram    = detect_cpu_ram_gb()
    models     = list_cached_models()

    if args.list or (not args.model and not sys.stdin.isatty()):
        print("已下载的模型：")
        for i, (mid, snap, cfg) in enumerate(models):
            disk = get_weight_size_gb(snap)
            cfg_str = "有 config" if cfg else "无 config"
            print(f"  [{i+1}] {mid:<45} {disk:5.1f} GB  ({cfg_str})")
        return

    # 确定目标模型
    if args.model:
        target = args.model
        matched = [(mid, snap, cfg) for mid, snap, cfg in models if mid == target]
        if not matched:
            print(f"[!] 在 HF 缓存中未找到 '{target}'，请先下载。")
            print(f"    缓存目录: {get_hf_cache_dir()}")
            sys.exit(1)
        model_id, snap_dir, config_path = matched[0]
    else:
        # 交互式选择
        print("已下载的模型：")
        for i, (mid, snap, cfg) in enumerate(models):
            disk = get_weight_size_gb(snap)
            cfg_str = "有 config" if cfg else "无 config"
            print(f"  [{i+1}] {mid:<45} {disk:5.1f} GB  ({cfg_str})")
        if not models:
            print("  (无已下载模型)")
            sys.exit(0)
        print()
        try:
            choice = int(input(f"选择模型 [1-{len(models)}]: ").strip())
            model_id, snap_dir, config_path = models[choice - 1]
        except (ValueError, IndexError):
            print("无效选择。")
            sys.exit(1)

    # GPU 列表（可被 --tp 限制）
    gpu_list = gpus if gpus else [{"name": "RTX 4090", "vram_gb": 24.0}] * 5

    analyze(model_id, snap_dir, config_path, args.dtype, gpu_list, cpu_ram)


if __name__ == "__main__":
    main()
