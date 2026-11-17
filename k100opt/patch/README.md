# 补丁目录

本目录是覆盖层的全部实现，不改动海光源码树。`sitecustomize.py` 在 Python 启动时注册一个 `MetaPathFinder`：被挂载的目标模块首次导入时，先执行原加载器，再调用对应补丁的 `install(module)` 完成替换。

```
patch/
├── sitecustomize.py     加载器与补丁注册表（须留在 PYTHONPATH 顶层，Python 启动时自动导入）
├── kpatch/              补丁实现包
│   ├── __init__.py      导出 PATCH_DIR，供包内定位内核 .so 等随包资源
│   ├── sched/           调度：预填充准入、块大小、首 token 送出顺序
│   ├── kv/              KV Cache 量化与自写分页注意力的接入
│   ├── ops/             算子：MoE、线性注意力、草稿选择
│   ├── comm/            通信：预填充 all-reduce 的 int8 分组量化
│   └── infra/           基础设施：启动参数、导入、预热、分词
└── libpa_*.so / libpp_*.so    自写内核编译产物（不入库，由 ../kernels/build.sh 生成，交付镜像内已含）
```

`start_server.sh` 把本目录加入 `PYTHONPATH`；`PYTHONPATH` 不含本目录时，任何补丁都不装载，行为与厂商默认一致。内核源码与编译见 `../kernels/`。各项的原理与效果见 `../docs/技术方案_最终版.md`（其附录 A 为补丁与开关总表，含两种平台上的取值），实现方式与核对、回退见 `../docs/实现说明.md`。

## R17（2026-09-22）的改动（历史记录；此后的改动见下方清单与 `git log`）

KV Cache FP8 量化专项落在下面这几个文件，其余补丁未改动。依据、正确性验证与性能数据见 `../docs/技术报告_KV量化专项_自写分页注意力与长请求准入.md`。

| 文件 | 状态 | 本轮做了什么 |
|---|---|---|
| `kpatch/kv/fp8_kv_hcu.py` | 新增 | 放行 e4m3 KV 池的两处拦截规则；把 decode（验证步 q=4、草稿步 q=1）与 prefill 的注意力改调自写内核，形状不符时反量化回退厂商内核 |
| `kpatch/kv/pa_py.py` | 新增 | 自写内核的 ctypes 包装；工作区按 `(bs, nsplit)` 缓存以固定地址，可被 CUDA graph 捕获 |
| `kpatch/sched/long_admit.py` | 新增 | 长请求准入：同时在场的长请求数超过 K 时，新到的长请求留在等待队列 |
| `../kernels/pa_v5.cpp`、`pp_v5.cpp`、`pa_ext.cpp` | 新增 | decode / prefill / 变长 q 三个自写分页注意力内核 |
| `sitecustomize.py` | 修改 | 登记上述三个补丁；补丁实现迁入 `kpatch` 包并按作用分子包 |
| `../start_server.sh` | 修改 | 增加 `--kv-cache-dtype fp8_e4m3` 与本专项的六个开关默认值 |

其余 17 个补丁文件本轮只有目录位置变化，内容与 R16 逐字相同（`git log --follow` 可核对）。

## 补丁清单

### `kpatch/sched/` 调度

| 补丁 | 环境开关（默认） | 挂载点 | 作用 |
|---|---|---|---|
| `adaptive_pmr.py` | `SGLANG_ADAPTIVE_PMR`（0） | `managers.schedule_policy` | 按请求长度决定一次预填充放几条：长请求逐条做，短请求合批 |
| `adaptive_chunk.py` | `SGLANG_ADAPTIVE_CHUNK`（0） | `managers.scheduler` | 按输入长度选预填充块大小并均分（≥32768 用 16384，≥65536 用 32768） |
| `first_token_first.py` | `SGLANG_FIRST_TOKEN_FIRST`（0） | `managers.scheduler` | 预填充结果先于下一批规划处理；首 token 不成字时先插一轮 decode |
| `long_admit.py` | `SGLANG_LONG_ADMIT`（0） | `managers.schedule_policy` | 限制同时在场的长请求数 ≤K，其余留在等待队列 |
| `pair_prefill.py` | `SGLANG_PAIR_PREFILL`（0；交付配置 1） | 由 `first_token_first` 与 `quant_ar` 导入 | 成对预填充：同一请求相邻块交错发射，一块的通信与另一块的计算重叠。成对期间 MoE 对齐排序改走 `ops/safe_align.py`（lightop 版在真并发下偶发丢 token） |

### `kpatch/kv/` KV Cache 量化与注意力内核

| 补丁 | 环境开关（默认） | 挂载点 | 作用 |
|---|---|---|---|
| `fp8_kv_hcu.py` | `SGLANG_FP8_KV_HCU`（0） | `server_args`、`arg_groups.overrides`、`attention.flashattention_interface` | 见上表 |
| `pa_py.py` | — | 被 `fp8_kv_hcu.py` 导入 | 见上表 |

配套开关：`SGLANG_PA_DIAG_BT`（页表结构诊断，**默认关且应保持关闭**——它在 decode 热路径上做一次设备同步，R17 的「首遍效应」即由此而来）、`SGLANG_PA_OWN`（decode 内核）、`SGLANG_PP_OWN`（prefill 内核）、`SGLANG_PA_FMT`（e4m3 / e5m2，须与 `--kv-cache-dtype` 一致）、`SGLANG_PA_EXT`（变长 q 变体，默认关）、`SGLANG_PA_TRACE`（记录注意力入口的实际路由，默认关）、`SGLANG_LONG_ADMIT_THRESH` 与 `SGLANG_LONG_ADMIT_K`（准入阈值与并发上限）。

### `kpatch/ops/` 算子

| 补丁 | 环境开关（默认） | 挂载点 | 作用 |
|---|---|---|---|
| `act_flat.py` | `SGLANG_ACT_FLAT`（1） | `moe_runner.triton_utils.fused_moe` | MoE 激活算子改为按行展平，仅在 M ≥ `SGLANG_ACT_FLAT_MIN` 时生效 |
| `fast_moe_align.py` | `SGLANG_FAST_MOE_ALIGN`（1） | 同上 | MoE 对齐跳过为专家并行准备的 5 个算子 |
| `lightop_compat.py` | `SGLANG_LIGHTOP_COMPAT`（1） | `moe_align_block_size` | 补上 `op.moe_align_block_size_out` 的兼容入口 |
| `fused_draft_topk.py` | `SGLANG_FUSED_DRAFT_TOPK`（1） | `speculative.eagle_worker_v2` | 草稿选择的 softmax 与 top-k 融合 |
| `mtp_rope_triton.py` | `SGLANG_MTP_ROPE_TRITON`（1） | `layers.rotary_embedding.mrope` | MTP 草稿层 1 维位置的 RoPE 扩成零步长 [3,T] 视图走 Triton 融合核（17 核→1 核）；只影响草稿数值 |
| `kv_write_fused.py` | `SGLANG_KV_WRITE_FUSED`（0） | `mem_cache.memory_pool` | 全注意力层 KV 写入（fp8 转换 + 分页写）6 核合 1，逐位一致；服务里并发 1 反而慢 4%（原因未明），默认关闭 |
| `vocab_par_draft.py` | `SGLANG_VPA_DRAFT`（1） | `layers.logits_processor`、`models.qwen3_5_mtp`、`speculative.eagle_worker_v2` | 草稿模型的 logits 不再全收集：各卡在本卡词表分片上求 (最大值, 首个下标, sumexp)，只收集 [M,3]；依赖 `fused_draft_topk` |
| `vocab_par_verify.py` | `SGLANG_VPA_VERIFY`（1） | `layers.logits_processor`、`speculative.eagle_worker_v2` | 验证步图内只算本卡分片与全局 argmax（写进共享 logits 缓冲末列）；带 logprob/语法/惩罚/logit_bias 的批次在图外补一次全收集后走原函数 |
| `gdn_nozero.py` | `SGLANG_GDN_NOZERO`（1） | `attention.fla.chunk` | 省去线性注意力链中一处无效清零（已用死缓冲区投毒法验证可删） |
| `gdn_cfg.py` | `SGLANG_GDN_CFG`（1） | `aiter…fused_sigmoid_gating_recurrent` | 为 gfx928 补线性注意力 decode 内核的配置（厂商配置表缺该架构） |
| `chunk_h_cfg.py` | `SGLANG_CHUNK_H_CFG`（1） | `aiter…chunk_delta_h` | 同上，预填充内核的配置 |
| `m3_fuse.py` | `SGLANG_M3_FUSE`（0） | `models.qwen2_moe` | MoE 归约与共享专家门控融合为一个内核 |
| `safe_align.py` | — | 被 `sched/pair_prefill.py` 导入 | 并发安全的 MoE 对齐排序（四个 Triton 内核），只在成对预填充期间替换 lightop 的 `moe_align_block_size` |
| `moe_pack_w13.py` | `SGLANG_MOE_PACK_W13`（0；风冷档位 1） | `layers.quantization.unquant`、`moe_runner.triton_utils.fused_moe` | 风冷专用：MoE 专家权重 w13 加载后就地重排为 tile 连续，内核改读打包布局；不支持打包布局的调用直接报错 |
| `conv1d_cfg.py` | `SGLANG_CONV1D_CFG`（0；风冷档位 1） | `layers.attention.mamba.causal_conv1d_triton` | 风冷专用：GDN 预填充因果卷积按本批总 token 数改写启动参数，输出逐位不变 |
| `dense_pack.py` | `SGLANG_DENSE_PACK`（0；风冷档位 1） | `layers.quantization.unquant` | 风冷专用：decode 稠密投影另存打包副本（每卡多占 620 MiB），M 在 8–64 时走打包 Triton 内核 |

### `kpatch/comm/` 通信

| 补丁 | 环境开关（默认） | 挂载点 | 作用 |
|---|---|---|---|
| `quant_ar.py` | `SGLANG_QUANT_AR`（0） | `distributed.parallel_state` | 预填充阶段的 all-reduce 改为 int8 分组量化传输。分组固定 32：128 会使超长上下文的接受长度下降 |

### `kpatch/infra/` 基础设施

| 补丁 | 环境开关（默认） | 挂载点 | 作用 |
|---|---|---|---|
| `server_defaults.py` | `SGLANG_SERVER_DEFAULTS`（1） | `server_args` | 未指定的启动参数自动补齐，关键环境变量缺失时告警 |
| `fast_import.py` | `SGLANG_FAST_IMPORT`（1） | `vllm_hcu.patch_utils` | 与厂商组件无关的模块走原生导入，缩短启动时间 |
| `warm_batch.py` | `SGLANG_WARM_BATCH`（1） | `entrypoints.warmup` | 补发 token 数为奇数的预热请求，把 Triton 即时编译放到服务就绪之前 |
| `tok_pool.py` | `SGLANG_TOK_POOL`（0） | `managers.tokenizer_manager` | 长文本分词进程池（逐位一致，失败时回退并记录原因） |

### 硬件档位

`sitecustomize.py` 的 `HW_PROFILE_ENV` 按 `SGLANG_HW_PROFILE` 改写少数环境变量：不设或 `liquid` 不改；`air`（`../start_server_air.sh`）打开 `kpatch/ops/` 表中的三项风冷专用补丁（`moe_pack_w13`、`conv1d_cfg`、`dense_pack`），并把 `PYTORCH_TUNABLEOP_FILENAME` 改指 `../config/tunableop/air/`，日志打印 `[hw_profile]` 一行。只改写未设置或仍等于液冷默认值的项。

## 新增补丁的做法

① 在 `kpatch/<组>/` 下建 `你的补丁.py`，实现 `install(module)`；需要定位随包资源时用 `from .. import PATCH_DIR`，不要用 `__file__` 推目录。

② 在 `sitecustomize.py` 的 `TARGETS` 中按「目标模块 → [(补丁模块全名, 开关名, 默认值)]」登记，补丁模块全名形如 `kpatch.sched.你的补丁`。

③ 补丁内打印两级日志：安装时的「已安装」与首次真正改变行为时的「首次真正生效」。只有前者而无后者一律视为未生效——静默回退比报错更危险。

④ 若涉及 CUDA graph，任何 `.item()` / `int()` 须排在 `is_current_stream_capturing()` 判定之后，否则会在图捕获期触发设备同步，导致捕获失败。

⑤ 跨补丁引用一律用包内相对导入（同组 `from . import x`，跨组 `from ..组名 import x`），不要写裸 `import x`。
