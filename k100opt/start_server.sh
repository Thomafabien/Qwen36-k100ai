#!/usr/bin/env bash
# 覆盖层启动脚本
# 所有路径可用环境变量覆盖，便于把仓库放到任意位置。
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3.6-35B-A3B}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/workspace/sglang-das/python}"
HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-8023}"

export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"
export PYTHONPATH="$REPO/patch:$SGLANG_PYTHON"
export SGLANG_MOE_CONFIG_DIR="${SGLANG_MOE_CONFIG_DIR:-$REPO/config/moe_config}"
export PYTORCH_TUNABLEOP_ENABLED="${PYTORCH_TUNABLEOP_ENABLED:-1}"
export PYTORCH_TUNABLEOP_TUNING="${PYTORCH_TUNABLEOP_TUNING:-0}"
export PYTORCH_TUNABLEOP_ROCBLAS_ENABLED="${PYTORCH_TUNABLEOP_ROCBLAS_ENABLED:-1}"
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED="${PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED:-0}"
export PYTORCH_TUNABLEOP_FILENAME="${PYTORCH_TUNABLEOP_FILENAME:-$REPO/config/tunableop/union.csv}"
export SGLANG_GDN_CFG="${SGLANG_GDN_CFG:-1}"
export SGLANG_CHUNK_H_CFG="${SGLANG_CHUNK_H_CFG:-1}"
export SGLANG_GDN_NOZERO="${SGLANG_GDN_NOZERO:-1}"
export SGLANG_FUSED_DRAFT_TOPK="${SGLANG_FUSED_DRAFT_TOPK:-1}"
export SGLANG_FAST_MOE_ALIGN="${SGLANG_FAST_MOE_ALIGN:-1}"
export SGLANG_ACT_FLAT="${SGLANG_ACT_FLAT:-1}"
export SGLANG_ACT_FLAT_MIN="${SGLANG_ACT_FLAT_MIN:-20000}"
export SGLANG_ADAPTIVE_PMR="${SGLANG_ADAPTIVE_PMR:-1}"
export SGLANG_ADAPTIVE_PMR_THRESH="${SGLANG_ADAPTIVE_PMR_THRESH:-4000}"
export SGLANG_USE_AITER_LINEAR_ATTN="${SGLANG_USE_AITER_LINEAR_ATTN:-1}"
export SGLANG_FAST_IMPORT="${SGLANG_FAST_IMPORT:-1}"
# 基线（R12, O 88.20）包含以下两项；2026-09-16 A/B 实测中性或略差，保留以保证与基线逐项一致
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
export SGLANG_M3_FUSE="${SGLANG_M3_FUSE:-1}"
# 2026-09-19 新增（各项依据见 patch/ 内对应文件头部）
export SGLANG_QUANT_AR="${SGLANG_QUANT_AR:-1}"                         # 预填充通信 int8 分组量化（含图内路径）
export SGLANG_QAR_GROUP="${SGLANG_QAR_GROUP:-32}"                       # 分组必须 32：128 在超长上下文下接受长度下降
export SGLANG_FIRST_TOKEN_FIRST="${SGLANG_FIRST_TOKEN_FIRST:-3}"        # 预填充结果先于下一批规划处理；首 token 不成字时先插一轮 decode
export SGLANG_ADAPTIVE_PMR_FIXCHUNK="${SGLANG_ADAPTIVE_PMR_FIXCHUNK:-1}" # 尾块不与下一条请求的首块拼批
export SGLANG_TOK_POOL="${SGLANG_TOK_POOL:-1}"                          # 长文本分词进程池（逐位一致）
export SGLANG_PAIR_PREFILL="${SGLANG_PAIR_PREFILL:-1}"          # 成对预填充（patch/kpatch/sched/pair_prefill.py）：同一请求相邻块交错发射，一块的通信与另一块的计算重叠
export SGLANG_PAIR_PREFILL_MOE_OVERLAP="${SGLANG_PAIR_PREFILL_MOE_OVERLAP:-1}"   # MoE 段也参与重叠；并发安全由成对期间改用 kpatch/ops/safe_align.py 保证（09-23 定位 lightop moe_align 的并发缺陷）
# 2026-09-24 新增：decode 每轮 4 次 logits 全收集（[M,248320] 走 PCIe）改为各卡本地求 argmax、只收集 [M,3]；输出逐位不变
export SGLANG_VPA_DRAFT="${SGLANG_VPA_DRAFT:-1}"     # 草稿 top1（patch/kpatch/ops/vocab_par_draft.py）
export SGLANG_VPA_VERIFY="${SGLANG_VPA_VERIFY:-1}"   # 验证 argmax（vocab_par_verify.py）；带 logprob/语法/惩罚/logit_bias 的批次图外补全收集，与原路径逐位相同
export SGLANG_MTP_ROPE_TRITON="${SGLANG_MTP_ROPE_TRITON:-1}"   # MTP 草稿层 RoPE 走与验证步相同的 Triton 核（mtp_rope_triton.py），只影响草稿数值
export SGLANG_ADAPTIVE_CHUNK="${SGLANG_ADAPTIVE_CHUNK:-1}"                # 长请求用大预填充块：≥32768 用 16384，≥65536 用 32768（patch/kpatch/sched/adaptive_chunk.py）
# 2026-09-22 KV 量化专项（各项依据见 docs/技术报告_KV量化专项_自写分页注意力与长请求准入.md）
export SGLANG_FP8_KV_HCU="${SGLANG_FP8_KV_HCU:-1}"        # 放行 fp8_e4m3 KV 池的两处闸门；prefill/decode 注意力换自写内核
export SGLANG_PA_OWN="${SGLANG_PA_OWN:-1}"                # decode（验证步 q=4 / 草稿步 q=1）自写分页注意力 v5
export SGLANG_PA_FMT="${SGLANG_PA_FMT:-e4m3}"             # KV 量化格式，须与 --kv-cache-dtype 一致（e4m3 精度 2× 于 e5m2）
export SGLANG_PP_OWN="${SGLANG_PP_OWN:-1}"                # prefill 自写分页注意力 v5（读 fp8 KV）
export SGLANG_LONG_ADMIT="${SGLANG_LONG_ADMIT:-1}"        # 长请求准入：在场长请求 ≤K，其余留队（patch/kpatch/sched/long_admit.py）
export SGLANG_LONG_ADMIT_THRESH="${SGLANG_LONG_ADMIT_THRESH:-65536}"   # 输入 ≥ 此 token 数视为长请求（32768 可再抬 P，但 O 下降）
export SGLANG_LONG_ADMIT_K="${SGLANG_LONG_ADMIT_K:-2}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"                          # KV cache 存储格式，须与 SGLANG_PA_FMT 一致（bf16 为 auto）

# MoE 调优表按驱动报告的设备名查找（文件名含 device_name=…）。设备名与 K500SM_AI 不同时表会静默落空，
# 这里按实际设备名复制一份（仅限同为 gfx928 的卡），并把结果打印出来。
python3 - <<'PYEOF' || true
import os, shutil, glob
try:
    import torch, triton
    d = os.path.join(os.environ["SGLANG_MOE_CONFIG_DIR"], "configs", "triton_" + triton.__version__.replace(".", "_"))
    dn = torch.cuda.get_device_name(0).replace(" ", "_"); arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    have = os.path.exists(os.path.join(d, f"E=256,N=128,device_name={dn}.json"))
    if have:
        print(f"[moe_cfg] 设备名 {dn}：调优表已就位")
    elif "gfx928" in arch and os.path.isdir(d):
        for src in glob.glob(os.path.join(d, "E=256,N=128,device_name=K500SM_AI*.json")):
            shutil.copy(src, src.replace("device_name=K500SM_AI", f"device_name={dn}"))
        print(f"[moe_cfg] ⚠ 设备名为 {dn}（不是 K500SM_AI），已按该名字复制调优表；架构 {arch}")
    else:
        print(f"[moe_cfg] ⚠ 设备名 {dn} / 架构 {arch} 无对应调优表，MoE 将回落默认配置（目录 {d}）")
except Exception as e:
    print(f"[moe_cfg] ⚠ 检查失败: {type(e).__name__}: {e}")
PYEOF

# 实验用的运行时开关文件：存在即静默关闭对应优化。正式启动前清掉，并把清了什么打印出来。
for f in /tmp/qar_off /tmp/qar_cfg.json /tmp/achunk_off /tmp/achunk_noeven /tmp/pair_off /tmp/pair_seq /tmp/long_admit_k /tmp/long_admit_thresh; do
  [ -e "$f" ] && { rm -f "$f"; echo "[switches] ⚠ 发现并已删除残留的实验开关文件 $f（它会静默关闭/改写一项优化）"; }
done
echo "[switches] $(env | grep -E '^SGLANG_|^PYTORCH_TUNABLEOP|^GPU_MAX_HW|^NCCL_' | sort | tr '\n' ' ')"

exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --tokenizer-path "$MODEL_PATH" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --tp 4 --disable-radix-cache --trust-remote-code \
  --host "$HOST" --port "$PORT" --disable-custom-all-reduce \
  --warmups prefill_shapes \
  --page-size 64 \
  --numa-node 0 1 2 3 \
  --prefill-max-requests 2 \
  --max-prefill-tokens 40960 \
  --kv-cache-dtype "$KV_DTYPE" \
  --cuda-graph-backend-prefill breakable \
  --cuda-graph-bs-prefill 4 8 16 32 64 128 256 512 768 1024 1088 1536 2048 2112 2176 2240 2304 2368 2432 \
  "$@"
