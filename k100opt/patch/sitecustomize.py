"""每个 Python 进程自动加载：在目标模块被导入后立刻打补丁。
父进程与所有 TP 子进程都覆盖到，且不改动海光源码树。
一个模块可挂多个补丁，按列表顺序安装；每个补丁有独立开关（=0 关闭）。

补丁实现位于同级的 `kpatch` 包，按作用分 sched / kv / ops / comm / infra 五个子包。
本文件须留在 `PYTHONPATH` 顶层：Python 启动时自动导入它，由它注册查找器。"""
import sys, os, importlib, importlib.abc

# 交付配置的环境变量：未设置的项在解释器启动时补上（已设置的不动），使「只要补丁目录在 PYTHONPATH 上，
# 无论用 start_server.sh 还是直接敲 launch_server 命令，得到的都是同一套配置」。与 start_server.sh 的 export 逐项一致
# （tools/check_env.py 会核对两边）。SGLANG_NO_DEFAULT_ENV=1 可关闭本段（做逐项对照实验时用）。
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIVERY_ENV = {
    "GPU_MAX_HW_QUEUES": "2",
    "SGLANG_MOE_CONFIG_DIR": os.path.join(_REPO, "config", "moe_config"),
    "PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "0", "PYTORCH_TUNABLEOP_ROCBLAS_ENABLED": "1",
    "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED": "0", "PYTORCH_TUNABLEOP_FILENAME": os.path.join(_REPO, "config", "tunableop", "union.csv"),
    "SGLANG_GDN_CFG": "1", "SGLANG_CHUNK_H_CFG": "1", "SGLANG_GDN_NOZERO": "1", "SGLANG_FUSED_DRAFT_TOPK": "1",
    "SGLANG_FAST_MOE_ALIGN": "1", "SGLANG_ACT_FLAT": "1", "SGLANG_ACT_FLAT_MIN": "20000",
    "SGLANG_ADAPTIVE_PMR": "1", "SGLANG_ADAPTIVE_PMR_THRESH": "4000", "SGLANG_ADAPTIVE_PMR_FIXCHUNK": "1",
    "SGLANG_USE_AITER_LINEAR_ATTN": "1", "SGLANG_FAST_IMPORT": "1", "NCCL_MIN_NCHANNELS": "16", "NCCL_MAX_NCHANNELS": "16",
    "SGLANG_M3_FUSE": "1", "SGLANG_QUANT_AR": "1", "SGLANG_QAR_GROUP": "32", "SGLANG_FIRST_TOKEN_FIRST": "3",
    "SGLANG_TOK_POOL": "1", "SGLANG_PAIR_PREFILL": "1", "SGLANG_PAIR_PREFILL_MOE_OVERLAP": "1", "SGLANG_ADAPTIVE_CHUNK": "1",
    # R17 起：KV FP8 与自写内核、长请求准入（--kv-cache-dtype 由 server_defaults 在未指定时补为 fp8_e4m3）
    "SGLANG_FP8_KV_HCU": "1", "SGLANG_PA_OWN": "1", "SGLANG_PA_FMT": "e4m3", "SGLANG_PP_OWN": "1",
    "SGLANG_LONG_ADMIT": "1", "SGLANG_LONG_ADMIT_THRESH": "65536", "SGLANG_LONG_ADMIT_K": "2",
    # R20 起：decode 的 logits 不再全收集（草稿 top1 与验证 argmax 改词表并行）
    "SGLANG_VPA_DRAFT": "1", "SGLANG_VPA_VERIFY": "1", "SGLANG_MTP_ROPE_TRITON": "1",
}
if os.environ.get("SGLANG_NO_DEFAULT_ENV", "0") == "0":
    _added = [k for k in DELIVERY_ENV if k not in os.environ]
    for _k in _added:
        os.environ[_k] = DELIVERY_ENV[_k]
    if _added and "sglang.launch_server" in " ".join(sys.argv) + " " + " ".join(getattr(sys, "orig_argv", [])):
        print(f"[delivery_env] 已补上 {len(_added)} 个未设置的环境变量: {' '.join(_added)}", flush=True)

# 硬件档位（2026-09-24，风冷适配）：风冷 K100_AI 板卡（VBIOS 6.x）的 DF 地址散列未开，跳行读的显存带宽只有液冷的 1/3–1/2，
# 需要四项只对风冷有效的改动（液冷单卡复测收益不足，不启用）：
#   MoE w13 预打包（kpatch/ops/moe_pack_w13.py）、风冷 GEMM 选解表（config/tunableop/air/）、
#   GDN 预填充卷积启动参数（kpatch/ops/conv1d_cfg.py）、decode 稠密投影打包（kpatch/ops/dense_pack.py）。
# MoE 调优表不需要切换：sglang 按设备名查表，风冷表存为 device_name=K100_AI，液冷为 K500SM_AI，各取各的。
# 显式指定、不自动识别（用户 09-24 定）：SGLANG_HW_PROFILE=air 才启用（风冷用 start_server_air.sh 启动即可）；
# 不设或设为 liquid 什么都不改，与原行为逐位相同。只改写「未设置或仍等于液冷默认值」的项，用户显式给的值一律不动。
HW_PROFILE_ENV = {
    "air": {"SGLANG_MOE_PACK_W13": "1",
            "PYTORCH_TUNABLEOP_FILENAME": os.path.join(_REPO, "config", "tunableop", "air", "union.csv"),
            "SGLANG_CONV1D_CFG": "1", "SGLANG_DENSE_PACK": "1"},
}
_hw = os.environ.get("SGLANG_HW_PROFILE", "liquid")
if os.environ.get("SGLANG_NO_DEFAULT_ENV", "0") == "0" and _hw in HW_PROFILE_ENV:
    _hw_set = []
    for _k, _v in HW_PROFILE_ENV[_hw].items():
        if os.environ.get(_k) in (None, DELIVERY_ENV.get(_k)):
            os.environ[_k] = _v; _hw_set.append(f"{_k}={_v}")
    if "sglang.launch_server" in " ".join(sys.argv) + " " + " ".join(getattr(sys, "orig_argv", [])):
        print(f"[hw_profile] 硬件档位 {_hw}：{' '.join(_hw_set) or '（均已显式设置，不改）'}", flush=True)

# 模块名 -> [(补丁模块, 开关环境变量, 开关默认值), ...]
TARGETS = {
    "sglang.srt.layers.attention.mamba.causal_conv1d_triton": [
        ("kpatch.ops.conv1d_cfg", "SGLANG_CONV1D_CFG", "0"),
    ],
    "sglang.srt.arg_groups.overrides": [
        ("kpatch.kv.fp8_kv_hcu", "SGLANG_FP8_KV_HCU", "0"),
    ],
    "sglang.srt.layers.attention.flashattention_interface": [
        ("kpatch.kv.fp8_kv_hcu", "SGLANG_FP8_KV_HCU", "0"),
    ],
    "vllm_hcu.patch_utils": [
        ("kpatch.infra.fast_import", "SGLANG_FAST_IMPORT", "1"),
    ],
    "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size": [
        ("kpatch.ops.lightop_compat", "SGLANG_LIGHTOP_COMPAT", "1"),
    ],
    "sglang.srt.speculative.eagle_worker_v2": [
        ("kpatch.ops.fused_draft_topk", "SGLANG_FUSED_DRAFT_TOPK", "1"),
        ("kpatch.ops.vocab_par_draft", "SGLANG_VPA_DRAFT", "1"),
        ("kpatch.ops.vocab_par_verify", "SGLANG_VPA_VERIFY", "1"),
    ],
    "sglang.srt.layers.logits_processor": [
        ("kpatch.ops.vocab_par_draft", "SGLANG_VPA_DRAFT", "1"),
        ("kpatch.ops.vocab_par_verify", "SGLANG_VPA_VERIFY", "1"),
    ],
    "sglang.srt.models.qwen3_5_mtp": [
        ("kpatch.ops.vocab_par_draft", "SGLANG_VPA_DRAFT", "1"),
    ],
    "sglang.srt.entrypoints.warmup": [
        ("kpatch.infra.warm_batch", "SGLANG_WARM_BATCH", "1"),
    ],
    "sglang.srt.models.qwen2_moe": [
        ("kpatch.ops.m3_fuse", "SGLANG_M3_FUSE", "0"),
    ],
    "sglang.srt.managers.schedule_policy": [
        ("kpatch.sched.adaptive_pmr", "SGLANG_ADAPTIVE_PMR", "0"),
        ("kpatch.sched.long_admit", "SGLANG_LONG_ADMIT", "0"),
    ],
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe": [
        ("kpatch.ops.moe_pack_w13", "SGLANG_MOE_PACK_W13", "0"),
        ("kpatch.ops.fast_moe_align", "SGLANG_FAST_MOE_ALIGN", "1"),
        ("kpatch.ops.act_flat", "SGLANG_ACT_FLAT", "1"),
    ],
    "sglang.srt.layers.attention.fla.chunk": [
        ("kpatch.ops.gdn_nozero", "SGLANG_GDN_NOZERO", "1"),
    ],
    "aiter.ops.triton.fla.fused_sigmoid_gating_recurrent": [
        ("kpatch.ops.gdn_cfg", "SGLANG_GDN_CFG", "1"),
    ],
    "aiter.ops.triton.fla.sglang.chunk_delta_h": [
        ("kpatch.ops.chunk_h_cfg", "SGLANG_CHUNK_H_CFG", "1"),
    ],
    "sglang.srt.server_args": [
        ("kpatch.infra.server_defaults", "SGLANG_SERVER_DEFAULTS", "1"),
        ("kpatch.kv.fp8_kv_hcu", "SGLANG_FP8_KV_HCU", "0"),
    ],
    "sglang.srt.managers.scheduler": [
        ("kpatch.sched.first_token_first", "SGLANG_FIRST_TOKEN_FIRST", "0"),
        ("kpatch.sched.adaptive_chunk", "SGLANG_ADAPTIVE_CHUNK", "0"),
    ],
    "sglang.srt.managers.tokenizer_manager": [
        ("kpatch.infra.tok_pool", "SGLANG_TOK_POOL", "0"),
    ],
    "sglang.srt.layers.quantization.unquant": [
        ("kpatch.ops.dense_pack", "SGLANG_DENSE_PACK", "0"),
        ("kpatch.ops.moe_pack_w13", "SGLANG_MOE_PACK_W13", "0"),
    ],
    "sglang.srt.mem_cache.memory_pool": [
        ("kpatch.ops.kv_write_fused", "SGLANG_KV_WRITE_FUSED", "0"),
    ],
    "sglang.srt.layers.rotary_embedding.mrope": [
        ("kpatch.ops.mtp_rope_triton", "SGLANG_MTP_ROPE_TRITON", "1"),
    ],
    "sglang.srt.distributed.parallel_state": [
        ("kpatch.comm.quant_ar", "SGLANG_QUANT_AR", "0"),
    ],
}

class _PatchLoader(importlib.abc.Loader):
    def __init__(self, real, plist):
        self.real, self.plist = real, plist
    def create_module(self, spec): return self.real.create_module(spec)
    def exec_module(self, module):
        self.real.exec_module(module)
        for pm, env, dflt in self.plist:
            if os.environ.get(env, dflt) == "0":
                continue
            try:
                importlib.import_module(pm).install(module)
            except Exception as e:
                print(f"[{pm.rsplit(chr(46), 1)[-1]}] 安装失败, 回退原实现: {type(e).__name__}: {e}", flush=True)

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        plist = TARGETS.get(fullname)
        if not plist: return None
        for f in sys.meta_path:
            if f is self: continue
            spec = f.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _PatchLoader(spec.loader, plist)
                return spec
        return None

if not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
