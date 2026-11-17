#!/usr/bin/env bash
# 风冷 K100_AI（VBIOS 6.x，DF 地址散列未开）用：打开风冷档（见 patch/sitecustomize.py 的 HW_PROFILE_ENV），其余与 start_server.sh 相同。
SGLANG_HW_PROFILE=air exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_server.sh" "$@"
