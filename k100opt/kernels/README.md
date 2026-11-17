# 自写分页注意力内核

本轮自研的两套 GPU 内核，由 `patch/kpatch/kv/fp8_kv_hcu.py` 在运行时以 ctypes 加载。交付镜像内已含编译好的 `.so`；本目录提供源码与编译方式，便于核对与修改。原理、正确性验证与性能数据见 `docs/技术报告_KV量化专项_自写分页注意力与长请求准入.md`。

## 文件

| 文件 | 用途 | 对应 `.so` |
|---|---|---|
| `pa_v5.cpp` | decode 分页注意力。`-DPA_SQ=4` 为验证步、`-DPA_SQ=1` 为草稿步；`-DQFMT=4` 为 e4m3、`=5` 为 e5m2 | `libpa_{e4m3,e5m2}_q{4,1}.so` |
| `pp_v5.cpp` | prefill 分页注意力 | `libpp_v5_q{4,5}.so` |
| `pa_ext.cpp` | 变长 q（≤4）的 split-KV 变体，默认不启用（`SGLANG_PA_EXT=1` 开启） | `libpa_ext_{e4m3,e5m2}.so` |
| `history/pa_lib.cpp`、`history/pa_v4.cpp` | decode 内核的历史版本，对应报告 5.2 节演进表 | — |
| `history/pp_v1.cpp`、`history/pp_v3.cpp`、`history/pp_v4.cpp` | prefill 内核的历史版本，对应报告 5.3 节演进表 | — |

## 编译

```bash
bash kernels/build.sh          # 一键编译全部变体到 patch/，并打印资源用量
```

等价的手工命令：

```bash
H=/opt/dtk/bin/hipcc
F="-O3 --offload-arch=gfx928 -ffp-contract=off -shared -fPIC"
cd kernels
for SQ in 4 1; do
  $H $F -DQFMT=4 -DPA_SQ=$SQ pa_v5.cpp -o ../patch/libpa_e4m3_q$SQ.so
  $H $F -DQFMT=5 -DPA_SQ=$SQ pa_v5.cpp -o ../patch/libpa_e5m2_q$SQ.so
done
$H $F -DQFMT=4 pp_v5.cpp -o ../patch/libpp_v5_q4.so
$H $F -DQFMT=5 pp_v5.cpp -o ../patch/libpp_v5_q5.so
$H $F -DQFMT=4 pa_ext.cpp -o ../patch/libpa_ext_e4m3.so
$H $F -DQFMT=5 pa_ext.cpp -o ../patch/libpa_ext_e5m2.so
```

查看资源用量（VGPR、溢出、LDS）：

```bash
$H -O3 --offload-arch=gfx928 -ffp-contract=off -S --cuda-device-only -DQFMT=4 -DPA_SQ=4 pa_v5.cpp -o pa_v5.s
grep -oE '(\.vgpr_count|\.vgpr_spill_count|\.group_segment_fixed_size): *[0-9]+' pa_v5.s
```

预期：`pa_v5` VGPR 107、无溢出、LDS 29952 B；`pp_v5` VGPR 162、无溢出、LDS 20736 B。

## 改动后必须重跑的检查

见 `docs/KV量化专项验证/说明.md`。最低限度：`01_正确性对拍/pa_mod64.py`（`seqlen mod 64` 全余数类与不等长 batch 的逐位对拍）与 `01_正确性对拍/longq.py`（服务内长上下文检索）。仅用整页长度对拍曾漏掉一处尾页同步缺陷，详见报告 5.1 节。
