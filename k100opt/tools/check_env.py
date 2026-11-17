#!/usr/bin/env python3
"""环境自检：把「换台机器就会踩」的坑一次查完。

跑基准之前先跑这个。每一项都标 PASS / WARN / FAIL，并写明后果。
  FAIL = 一定跑不出预期结果，必须修
  WARN = 可能静默降级，建议核对

用法:  PYTHONPATH=<repo>/patch:/workspace/sglang-das/python python3 tools/check_env.py
"""
import os, sys, json, importlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = {"PASS": 0, "WARN": 0, "FAIL": 0}


def say(level, title, detail=""):
    R[level] += 1
    mark = {"PASS": "  ✓", "WARN": "  ! ", "FAIL": "  ✗"}[level]
    print(f"{mark} [{level}] {title}")
    if detail:
        for line in detail.splitlines():
            print(f"        {line}")


# 本仓库验证过的版本（K500SM_AI ×4 / DTK 26.04）
PINNED = {
    "torch":        "2.10.0+das.opt1.dtk2604.2607011430.gd27280",
    "triton":       "3.6.0+dtk2604.torch2100.2607031022.g5f9350",
    "aiter":        "0.1.3+das.opt1.dtk2604.torch2100.2607031825.g1e61ff",
    "lightop":      "0.6.0+das.dtk2604.torch2100.2608111356.g0ad03b",
    "vllm":         "0.18.1+das.dtk2604.torch2100.2606041655.ge5a62f",
    "vllm_hcu":     "0.18.1+das.dtk2604.torch2100.2607021911.gabf4af",
}
SGLANG_DAS_COMMIT = "93b80182"
SGLANG_DAS_REPO = "https://github.com/HYGON-AI/sglang-das.git"

print("=" * 68)
print("环境自检 —— K100AI Qwen3.6 推理优化覆盖层")
print("=" * 68)

# ---------- 1. 包版本 ----------
print("\n[1] 依赖版本（不符不一定坏，但结果不可与本仓库记录的成绩相比）")
try:
    from importlib.metadata import version as _v
except ImportError:
    from importlib_metadata import version as _v
for pkg, want in PINNED.items():
    try:
        got = _v(pkg)
    except Exception:
        say("FAIL", f"{pkg} 未安装", f"应为 {want}")
        continue
    if got == want:
        say("PASS", f"{pkg} = {got}")
    else:
        say("WARN", f"{pkg} 版本不同", f"当前 {got}\n本仓库验证过 {want}")

# ---------- 2. sglang 到底用的哪一份 ----------
print("\n[2] sglang 来源（最容易踩：pip 装的那份会被 PYTHONPATH 压过）")
try:
    import sglang
    path = os.path.dirname(sglang.__file__)
    if "sglang-das" in path:
        say("PASS", "sglang 解析到源码树", path)
    else:
        say("FAIL", "sglang 用的是 pip 包，不是源码树",
            f"当前 {path}\n必须: git clone {SGLANG_DAS_REPO}\n"
            f"      git checkout {SGLANG_DAS_COMMIT}\n"
            f"      PYTHONPATH=<repo>/patch:<sglang-das>/python")
except Exception as e:
    say("FAIL", "sglang 导入失败", str(e))

# ---------- 3. lightop 子模块与符号位置 ----------
print("\n[3] lightop 子模块与符号位置")
# 3a 子模块完整性：sglang-das 会 import lightop.quant 等，旧版本缺这些子包
for sub, why in [("quant", "sglang/kernels/ops/quantization/fp8_kernel.py:64 要 lightop.quant.per_token_quant_fp8"),
                 ("moe", "moe_align_block_size_out 在此"),
                 ("activation", ""), ("attention", ""), ("norm", ""), ("gemm_ops", "")]:
    try:
        importlib.import_module(f"lightop.{sub}")
        say("PASS", f"lightop.{sub} 存在")
    except Exception as e:
        say("FAIL", f"lightop.{sub} 缺失 —— lightop 版本过旧",
            (why + "\n" if why else "") +
            f"报错：{type(e).__name__}: {e}\n"
            "修法（注意必须选 torch2100 分支）：\n"
            f"  pip install \"lightop=={PINNED['lightop']}\" \\\n"
            "    --index-url https://pypi.sourcefind.cn/customer/dtk2604-rc4/simple")
try:
    from lightop import op as _op
    in_op = hasattr(_op, "moe_align_block_size_out")
    try:
        from lightop import moe as _moe
        in_moe = hasattr(_moe, "moe_align_block_size_out")
    except ImportError:
        in_moe = False
    if in_op:
        say("PASS", "moe_align_block_size_out 在 lightop.op（无需补丁）")
    elif in_moe:
        say("PASS", "moe_align_block_size_out 在 lightop.moe（由 lightop_compat 补丁处理）")
    else:
        say("FAIL", "两处都找不到 moe_align_block_size_out",
            "MoE 会在运行时 AttributeError。请核对 lightop 版本。")
except Exception as e:
    say("FAIL", "lightop 导入失败", str(e))

# ---------- 3.5 整条导入链（一次抓完所有缺包/缺符号）----------
print("\n[3.5] sglang-das 导入链（最重要：一次性暴露所有缺包与缺符号）")
try:
    from sglang.srt.server_args import prepare_server_args  # noqa: F401
    say("PASS", "prepare_server_args 导入成功",
        "整条链走通（含 lightop / sgl_kernel / vllm_hcu / causal-conv1d 等）")
except Exception as e:
    import traceback
    tb = traceback.format_exc().strip().splitlines()
    tail = "\n".join(tb[-6:])
    say("FAIL", f"导入链断在 {type(e).__name__}",
        tail + "\n\n修法：按 env/requirements_das.txt 装齐全部 das 包（20 个），"
        "\n不要只装报错提到的那一个——缺一个就会报下一个：\n"
        "  pip install --index-url https://pypi.sourcefind.cn/customer/dtk2604-rc4/simple \\\n"
        "      -r env/requirements_das.txt")

# ---------- 4. TunableOp 选解表 Validator ----------
print("\n[4] TunableOp 选解表（不匹配 → 整张表被拒 → 约值 10 个 P 点）")
tab = os.environ.get("PYTORCH_TUNABLEOP_FILENAME", os.path.join(REPO, "config/tunableop/union.csv"))
base, ext = os.path.splitext(tab)
tab0 = f"{base}0{ext}"          # PyTorch 会自动追加设备序号
if not os.path.exists(tab0):
    say("FAIL", "选解表不存在", f"找不到 {tab0}\n（注意 PyTorch 会在文件名后追加设备序号）")
else:
    want = {}
    for line in open(tab0):
        p = line.strip().split(",")
        if len(p) >= 3 and p[0] == "Validator":
            want[p[1]] = p[2]
    try:
        import torch
        cur = {
            "PT_VERSION": torch.__version__.split("+")[0],
            "GCN_ARCH_NAME": torch.cuda.get_device_properties(0).gcnArchName,
        }
        bad = [k for k, v in cur.items() if k in want and want[k] != v]
        if bad:
            say("FAIL", "选解表 Validator 不匹配（表会被静默拒绝）",
                "\n".join(f"{k}: 表={want[k]}  当前={cur[k]}" for k in bad) +
                "\n后果：所有 GEMM 退回默认启发式，P 列约掉 10 点，只在日志里留一条 UserWarning")
        else:
            say("PASS", f"Validator 可比项匹配（表 {len(want)} 项，已核 {len(cur)} 项）",
                "注意 HIPBLASLT_VERSION / ROCBLAS_VERSION 含构建哈希，本脚本无法比对，\n"
                "若换了 DTK 构建请留意服务日志里的 'Failed validator' 警告")
    except Exception as e:
        say("WARN", "无法读取当前环境信息比对", str(e))

# ---------- 5. MoE 配置目录名与 triton 自报版本 ----------
print("\n[5] MoE 配置目录（目录名由 triton.__version__ 拼出，不符则静默回落默认）")
try:
    import triton
    tv = triton.__version__.replace(".", "_")
    want_dir = os.path.join(REPO, "config/moe_config/configs", f"triton_{tv}")
    have = os.path.isdir(want_dir)
    pkg_v = None
    try:
        pkg_v = _v("triton")
    except Exception:
        pass
    note = ""
    if pkg_v and not pkg_v.startswith(triton.__version__):
        note = f"\n注意：pip 包版本 {pkg_v} 与模块自报 {triton.__version__} 不一致，这是正常的，以模块自报为准"
    if have:
        say("PASS", f"配置目录存在 triton_{tv}", note.strip())
        # 调优表按「驱动报告的设备名」查找：名字不同即静默回落默认配置（MoE 慢约 2.4 倍）
        try:
            import torch
            dn = torch.cuda.get_device_name(0).replace(" ", "_")
            miss = [f for f in (f"E=256,N=128,device_name={dn}.json", f"E=256,N=128,device_name={dn}_down.json")
                    if not os.path.exists(os.path.join(want_dir, f))]
            if miss:
                say("FAIL", f"设备名 {dn} 没有对应的 MoE 调优表", f"缺 {miss}\n"
                    f"现有: {sorted(os.listdir(want_dir))}\n"
                    "start_server.sh 会在启动时按实际设备名自动复制一份（同为 gfx928 架构时适用）；"
                    "若仓库目录只读，请手工复制 device_name=K500SM_AI*.json 并改名")
            else:
                say("PASS", f"设备名 {dn} 的 MoE 调优表存在（含 _down）")
        except Exception as e:
            say("WARN", "设备名检查失败", str(e))
    else:
        say("FAIL", f"缺少配置目录 triton_{tv}",
            f"仓库里现有: {os.listdir(os.path.join(REPO,'config/moe_config/configs'))}\n"
            f"后果：MoE tile 配置静默回落海光默认（本仓库单项最大收益失效）{note}")
except Exception as e:
    say("WARN", "triton 检查失败", str(e))

# ---------- 6. 硬件 ----------
print("\n[6] 硬件")
try:
    import torch
    n = torch.cuda.device_count()
    arch = torch.cuda.get_device_properties(0).gcnArchName
    name = torch.cuda.get_device_properties(0).name
    (say("PASS", f"{n} 张卡  {name}  {arch}") if n == 4 and "gfx928" in arch
     else say("WARN", f"{n} 张卡  {name}  {arch}", "本仓库在 4×gfx928(K500SM_AI) 上验证"))
except Exception as e:
    say("FAIL", "GPU 不可用", str(e))

# ---------- 6.5 硬件档位 ----------
# 风冷平台（驱动设备名 K100_AI）须以 SGLANG_HW_PROFILE=air（start_server_air.sh）启动；不设时是液冷档位，
# MoE 为未打包内核配打包模式分块表，成绩未实测。只按设备名判断，不读驱动参数文件。
print("\n[6.5] 硬件档位（风冷平台须用 start_server_air.sh，即 SGLANG_HW_PROFILE=air）")
try:
    import torch
    _dn = torch.cuda.get_device_properties(0).name.replace(" ", "_")
    _hp = os.environ.get("SGLANG_HW_PROFILE", "")
    if "K100_AI" in _dn and _hp != "air":
        say("WARN", f"设备名 {_dn}（风冷平台），当前 SGLANG_HW_PROFILE={_hp or '（未设，液冷档位）'}",
            "风冷平台请用 start_server_air.sh 启动，或给 bench/run_all.sh 加 SGLANG_HW_PROFILE=air 与 HIP_VISIBLE_DEVICES；\n"
            "若本次只是自检、随后会用 start_server_air.sh 启动，可忽略本条")
    elif "K500SM_AI" in _dn and _hp == "air":
        say("WARN", f"设备名 {_dn}（液冷平台），但 SGLANG_HW_PROFILE=air", "液冷平台不应打开风冷档位")
    else:
        say("PASS", f"设备名 {_dn}，SGLANG_HW_PROFILE={_hp or '（未设，液冷档位）'}")
except Exception as e:
    say("WARN", "硬件档位检查失败", str(e))

# ---------- 7. 必须的环境变量 ----------
print("\n[7] 关键环境变量（start_server.sh 会设，单独跑时需自己设）")
for k, want, why in [("GPU_MAX_HW_QUEUES", "2", "共享机器上他人队列会使 runlist 超额导致挂起")]:
    got = os.environ.get(k)
    (say("PASS", f"{k}={got}") if got == want
     else say("WARN", f"{k} 未设或不为 {want}", f"当前 {got!r}；{why}"))

# ---------- 8. 两条启动路径的配置一致性（静态，不需要 GPU）----------
# start_server.sh 的 export 与 patch/sitecustomize.py 的 DELIVERY_ENV 必须逐项一致，否则
# 「直接敲 launch_server」会静默得到另一套配置。R17 曾漏掉 7 个开关：直接启动时 KV 回到 bf16、
# 长请求准入关闭，P 由约 150 退回约 107，且不报任何错。
print("\n[8] 启动路径一致性（start_server.sh 对 sitecustomize.DELIVERY_ENV / server_defaults）")
try:
    import re, ast
    ss = open(os.path.join(REPO, "start_server.sh"), encoding="utf-8").read()
    exp = dict(re.findall(r'export (\w+)="\$\{\w+:-([^}]*)\}"', ss))
    src = open(os.path.join(REPO, "patch", "sitecustomize.py"), encoding="utf-8").read()
    body = re.search(r"DELIVERY_ENV = (\{.*?\n\})", src, re.S).group(1)
    body = re.sub(r"os\.path\.join\(_REPO, ([^)]*)\)",
                  lambda m: repr("<REPO>/" + "/".join(ast.literal_eval("[" + m.group(1) + "]"))), body)
    denv = ast.literal_eval(body)
    only_ss = sorted(set(exp) - set(denv)); only_d = sorted(set(denv) - set(exp))
    diff = [(k, exp[k], denv[k]) for k in sorted(set(exp) & set(denv))
            if exp[k] != denv[k] and not denv[k].startswith("<REPO>/")]
    if only_ss or only_d or diff:
        say("FAIL", "两条启动路径的环境变量不一致（直接敲 launch_server 会得到另一套配置）",
            (f"start_server.sh 有、DELIVERY_ENV 没有：{only_ss}\n" if only_ss else "")
            + (f"DELIVERY_ENV 有、start_server.sh 没有：{only_d}\n" if only_d else "")
            + (f"默认值不同：{diff}" if diff else ""))
    else:
        say("PASS", f"start_server.sh 与 DELIVERY_ENV 的 {len(exp)} 个环境变量逐项一致")
    kv_ss = re.search(r'KV_DTYPE="\$\{KV_DTYPE:-([^}]*)\}"', ss).group(1)
    sd = open(os.path.join(REPO, "patch", "kpatch", "infra", "server_defaults.py"), encoding="utf-8").read()
    kv_sd = "fp8_e4m3" if ('"--kv-cache-dtype", "fp8_e4m3"' in sd and denv.get("SGLANG_FP8_KV_HCU") == "1") else "（未补）"
    (say("PASS", f"KV 格式默认值一致：{kv_ss}") if kv_ss == kv_sd
     else say("FAIL", "KV 格式默认值不一致", f"start_server.sh 为 {kv_ss}，直接启动时 server_defaults 补的是 {kv_sd}"))
except Exception as e:
    say("FAIL", "一致性检查本身失败", f"{type(e).__name__}: {e}")

# CPU 调频策略：短输入 TTFT 受 CPU 发射速度限制，schedutil 下慢 10–25%（全表 O 约 −5）。服务启动时会尝试自动设为 performance。
try:
    import glob as _g
    _gov = sorted({open(f).read().strip() for f in _g.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor")})
    if _gov == ["performance"]:
        say("PASS", "CPU 调频策略 performance")
    elif _gov:
        say("WARN", f"CPU 调频策略为 {'/'.join(_gov)}：短输入 TTFT 慢 10–25%（全表 O 约 −5）",
            "服务启动时会尝试自动改为 performance（需写 /sys 的权限）；或在主机执行：\n"
            "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $f; done")
    else:
        say("WARN", "读不到 CPU 调频设置，无法确认（短输入 TTFT 对调频策略敏感）")
except Exception as e:
    say("WARN", "CPU 调频检查失败", str(e))

# GPU 性能档位：manual 低档位（液冷机曾被固定在 sclk 1270 MHz，最高 1400）下 decode 约慢 3%。服务启动时会尝试自动设为 auto。
try:
    import subprocess as _sp, re as _re
    _o = _sp.run(["hy-smi", "-p"], capture_output=True, text=True, timeout=15).stdout
    _lv = sorted(set(_re.findall(r"Performance Level:\s*(\S+)", _o)))
    if _lv == ["auto"]:
        say("PASS", "GPU 性能档位 auto")
    elif _lv:
        _c = sorted(set(_re.findall(r"sclk clock level: \d+ \((\d+)Mhz\)", _sp.run(["hy-smi", "-c"], capture_output=True, text=True, timeout=15).stdout)))
        say("WARN", f"GPU 性能档位为 {'/'.join(_lv)}（当前 sclk {'/'.join(_c)} MHz）：decode 约慢 3%",
            "服务启动时会尝试自动改为 auto（负载时升到最高档、空闲降频）；或在主机执行：hy-smi --setperflevel auto")
    else:
        say("WARN", "读不到 GPU 性能档位（hy-smi -p 无输出）")
except Exception as e:
    say("WARN", "GPU 性能档位检查失败", str(e))

# 成对预填充依赖 greenlet；缺失时它会自行停用（不报错），长行 TTFT 回退约 10%
try:
    import greenlet  # noqa: F401
    say("PASS", f"greenlet {getattr(greenlet, '__version__', '?')}（成对预填充依赖）")
except Exception:
    say("FAIL", "缺 greenlet —— 成对预填充会自行停用，长行 TTFT 回退约 10%", "pip install greenlet")

print("\n" + "=" * 68)
print(f"结果：{R['PASS']} PASS   {R['WARN']} WARN   {R['FAIL']} FAIL")
print("=" * 68)
sys.exit(1 if R["FAIL"] else 0)
