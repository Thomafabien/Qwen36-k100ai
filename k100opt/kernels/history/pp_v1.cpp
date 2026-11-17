// 自写分页 prefill 注意力（读 fp8 KV）：与 decode 内核 pa_lib.cpp 同一套部件，wave 分工反过来——
//   decode：4 个 wave 沿 KV token 维分工，Q 是全 block 共享的 16 行
//   prefill：4 个 wave 沿 Q 维分工（各持 4 token × 4 头 = 16 行），共享同一 KV 页
// 一页 K/V 读进 LDS 后服务 64 行 Q ⇒ L2→CU 流量 = 厂商(BlockM 64、4 头各读、bf16) 的 1/8。
// grid = (chunk/16 个 Q 块, bs)，block 256 线程；无 split、无 combine。
#include <hip/hip_runtime.h>
#include <cmath>
using u8 = unsigned char; using u16 = unsigned short;
typedef _Float16 half_t;
typedef half_t half4 __attribute__((ext_vector_type(4)));
typedef float  f32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ u16 f2bf(float f){ unsigned u=__float_as_uint(f); if(!(f==f)) return 0x7fc0; return (u16)((u + 0x7fffu + ((u>>16)&1u))>>16); }
__device__ __forceinline__ float fexp(float x){ return __builtin_amdgcn_exp2f(x * __uint_as_float(0x3fb8aa3bu)); }
__device__ __forceinline__ unsigned shfl_lane(unsigned v, int src){ return __builtin_amdgcn_ds_bpermute(src << 2, v); }
__device__ __forceinline__ float bf2fl(u16 h) { return __uint_as_float((unsigned)h << 16); }

#define HD   256
#define PAGE 64
#define HQ   4
#define NW   4              // 每 block 4 个 wave
#define TPW  4              // 每 wave 4 个 query token（× HQ 头 = 16 行）
#define TPB  (NW * TPW)     // 每 block 16 个 query token
#ifndef QFMT
#define QFMT 4              // 4=e4m3  5=e5m2
#endif

#if QFMT == 5
__device__ __forceinline__ half4 e5m2x4(unsigned w) {
    half4 r;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        r[i] = __builtin_bit_cast(half_t, (u16)((u16)((w >> (8 * i)) & 0xFF) << 8));
    return r;
}
#define QDEQ(w) e5m2x4(w)
#define QCORR   1.0f
#else
// e4m3：位重排后规格数与次规格数都恰好差 2^8，由 scale 与最终归一化各吸收一次（见 pa_lib.cpp）
__device__ __forceinline__ half4 e4m3x4(unsigned w) {
    const unsigned ev = w & 0x00FF00FFu;
    const unsigned od = (w >> 8) & 0x00FF00FFu;
    const unsigned fe = ((ev & 0x00800080u) << 8) | ((ev & 0x007F007Fu) << 7);
    const unsigned fo = ((od & 0x00800080u) << 8) | ((od & 0x007F007Fu) << 7);
    half4 r;
    r[0] = __builtin_bit_cast(half_t, (u16)(fe & 0xFFFFu));
    r[1] = __builtin_bit_cast(half_t, (u16)(fo & 0xFFFFu));
    r[2] = __builtin_bit_cast(half_t, (u16)(fe >> 16));
    r[3] = __builtin_bit_cast(half_t, (u16)(fo >> 16));
    return r;
}
#define QDEQ(w) e4m3x4(w)
#define QCORR   256.0f
#endif

// LDS 行宽 68 B = 17 dword：与 32 互质，半 wave 的 16 个 L 行落在不同 bank（与 decode 内核同）
#define ROWB 68

extern "C" __global__ __launch_bounds__(256)
void pp_v1(u16* __restrict__ out,               // [total_q, HQ, HD] bf16
           const u16* __restrict__ q,           // [total_q, HQ, HD] bf16
           const u8* __restrict__ kc,           // [pages, 1, PAGE, HD] fp8   K[token][dim]
           const u8* __restrict__ vc,           // [pages, 1, HD, PAGE] fp8   V[dim][token]
           const int* __restrict__ bt,          // [bs, n_per]
           const int* __restrict__ cu_q,        // [bs+1]
           const int* __restrict__ slen,        // [bs]  KV 总长（前缀 + 本 chunk）
           int n_per, float scale) {
    const int qb = blockIdx.x, b = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 63, w = tid >> 6;
    const int L = lane & 15, H = lane >> 4;
    const int q0 = cu_q[b], qlen = cu_q[b + 1] - q0;
    if (qb * TPB >= qlen) return;                       // 整 block 无效（所有线程一起退出，无 sync 悬挂）
    const int seqlen = slen[b];
    const int prefix = seqlen - qlen;
    const int tok_local = qb * TPB + w * TPW + (L / HQ);     // 本 lane 的 query（chunk 内序号）
    const bool qvalid = tok_local < qlen;
    const int qpos = prefix + (qvalid ? tok_local : 0);      // 绝对位置（因果用）
    const int qmax = prefix + min(qb * TPB + TPB - 1, qlen - 1);
    const int npages = qmax / PAGE + 1;                      // 因果：只需扫到含 qmax 的页

    // ── Q 常驻寄存器：lane(L,H) 持 Q[row L][dim 4H+i+16kk]；row L = (token tok_local, 头 L%HQ)
    half4 qreg[16];
    {
        const u16* qp = q + ((size_t)(q0 + (qvalid ? tok_local : 0)) * HQ + (L % HQ)) * HD;
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            half4 t;
#pragma unroll
            for (int i = 0; i < 4; ++i) t[i] = qvalid ? (half_t)bf2fl(qp[kk * 16 + 4 * H + i]) : (half_t)0.0f;
            qreg[kk] = t;
        }
    }

    float m_run = -INFINITY, l_run = 0.f;
    f32x4 acc[16];
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) acc[jj] = (f32x4){0.f, 0.f, 0.f, 0.f};

    // K 阶段 [4 段][64 token][68 B]，V 阶段 [256 dim][68 B]：同为 17408 B，复用一块
    __shared__ __align__(16) u8 s_kv[HD * ROWB];
    __shared__ half_t s_p[NW][16][18];                  // P 重排（行 9 dword，半 wave 无冲突）

    for (int p = 0; p < npages; ++p) {
        const int pg = bt[b * n_per + p];
        const int ntok = min(PAGE, seqlen - p * PAGE);
        const u8* kp = kc + (size_t)pg * PAGE * HD;
        const u8* vp = vc + (size_t)pg * HD * PAGE;

        // ── ① 装 K 页：线程 t 读 token t/4 的 dim 段 (t%4)·64..+63（64 B 连续）→ s_k[段][token][·]
        __syncthreads();
        {
            const int tk = tid >> 2, sg = tid & 3;
            const u8* src = kp + (size_t)tk * HD + sg * 64;
            u8* dst = s_kv + (sg * PAGE + tk) * ROWB;
#pragma unroll
            for (int c = 0; c < 4; ++c) *(u32x4*)&dst[c * 16] = *(const u32x4*)(src + c * 16);
        }
        __syncthreads();

        // ── ② S = Q·Kᵀ，页内 4 个 16-token 块；s[tb][j] = S[row L][token 16tb + H + 4j]
        f32x4 s[4];
#pragma unroll
        for (int tb = 0; tb < 4; ++tb) {
            f32x4 acc_s = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int r4 = 0; r4 < 4; ++r4) {
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    // B[4H+i][L] = K[token 16tb+L][dim 64r4 + 16kk + 4H + i]
                    const unsigned raw = *(const unsigned*)&s_kv[(r4 * PAGE + 16 * tb + L) * ROWB + 16 * kk + 4 * H];
                    acc_s = __builtin_amdgcn_mmac_f32_16x16x16f16(qreg[r4 * 4 + kk], QDEQ(raw), acc_s);
                }
            }
            s[tb] = acc_s;
        }

        // ── ③ 在线 softmax，按整页更新一次：掩码（页内有效 + 因果）→ 页内行最大 → 校正
        float mb = -INFINITY;
#pragma unroll
        for (int tb = 0; tb < 4; ++tb)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int tok = 16 * tb + H + 4 * j;
                const int gtok = p * PAGE + tok;
                float v = (tok < ntok && gtok <= qpos) ? s[tb][j] * scale : -INFINITY;
                s[tb][j] = v; mb = fmaxf(mb, v);
            }
#pragma unroll
        for (int d = 16; d <= 32; d <<= 1)
            mb = fmaxf(mb, __uint_as_float(shfl_lane(__float_as_uint(mb), lane ^ d)));
        const float m_new = fmaxf(m_run, mb);
        const float corr = (m_run == -INFINITY) ? 0.f : fexp(m_run - m_new);
        float lb = 0.f;
#pragma unroll
        for (int tb = 0; tb < 4; ++tb)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                s[tb][j] = (s[tb][j] == -INFINITY) ? 0.f : fexp(s[tb][j] - m_new);
                lb += s[tb][j];
            }
#pragma unroll
        for (int d = 16; d <= 32; d <<= 1)
            lb += __uint_as_float(shfl_lane(__float_as_uint(lb), lane ^ d));
        l_run = l_run * corr + lb;
        m_run = m_new;
#pragma unroll
        for (int jj = 0; jj < 16; ++jj)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[jj][j] *= corr;

        // ── ④ 装 V 页（转置存 [dim][token]）：线程 t 读 dim t 的 64 个 token（64 B 连续）
        __syncthreads();                                     // K 读完才能覆盖
#pragma unroll
        for (int c = 0; c < 4; ++c)
            *(u32x4*)&s_kv[tid * ROWB + c * 16] = *(const u32x4*)(vp + (size_t)tid * PAGE + c * 16);
        __syncthreads();

        // ── ⑤ O += P·V：每个 token 块先把 P 从 C 布局重排成 A 布局，再 16 条 mmac
#pragma unroll
        for (int tb = 0; tb < 4; ++tb) {
#pragma unroll
            for (int j = 0; j < 4; ++j) s_p[w][L][H + 4 * j] = (half_t)s[tb][j];
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();                    // wave 内可见即可；barrier 仅作 LDS 顺序点
            half4 pa;
#pragma unroll
            for (int i = 0; i < 4; ++i) pa[i] = s_p[w][L][4 * H + i];
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();                    // 下一 tb 覆写 s_p 前，所有 lane 已读完
            const int t0 = 16 * tb;
#pragma unroll
            for (int jj = 0; jj < 16; ++jj) {
                // B[4H+i][L] = V[token t0+4H+i][dim 16jj+L]；LDS 里 s_vt[dim][token]
                const unsigned raw = *(const unsigned*)&s_kv[(16 * jj + L) * ROWB + t0 + 4 * H];
                acc[jj] = __builtin_amdgcn_mmac_f32_16x16x16f16(pa, QDEQ(raw), acc[jj]);
            }
        }
    }

    // ── 归一化并经 LDS 转置成行连续后合并写出（复用 s_kv，分 2 批 × 128 dim；每 wave 16 行 × 128 × 2 B = 4 KB）
    const float inv = (l_run > 0.f) ? (QCORR / l_run) : 0.f;
    u16* op = out + ((size_t)(q0 + (qvalid ? tok_local : 0)) * HQ + (L % HQ)) * HD;
    for (int batch = 0; batch < 2; ++batch) {
        __syncthreads();
        u16* so = (u16*)s_kv + w * (16 * 136);              // 行 136 half = 272 B = 68 dword（写侧 2 路冲突，可接受）
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int jj = batch * 8 + t;
#pragma unroll
            for (int j = 0; j < 4; ++j) so[L * 136 + H + 4 * j + 16 * t] = f2bf(acc[jj][j] * inv);
        }
        __syncthreads();
        // 写出：本 wave 的 16 行各 128 个 bf16；lane 一次搬 2 个（4 B），64 lane 覆盖一行的 128 个
        // 行 r 对应的全局指针要按 r 的 query 算，用 bpermute 从 lane r 取 op 的低/高 32 位
        const unsigned op_lo = (unsigned)((size_t)op & 0xffffffffu), op_hi = (unsigned)((size_t)op >> 32);
        const unsigned qv = qvalid ? 1u : 0u;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const unsigned lo = shfl_lane(op_lo, r), hi = shfl_lane(op_hi, r), v = shfl_lane(qv, r);
            if (v) {
                u16* dst = (u16*)(((size_t)hi << 32) | lo) + batch * 128 + lane * 2;
                *(unsigned*)dst = *(const unsigned*)&so[r * 136 + lane * 2];
            }
        }
    }
}

extern "C" {
void pp_prefill(void* out, const void* q, const void* kc, const void* vc,
                const void* bt, const void* cu_q, const void* slen,
                int bs, int n_per, int max_qlen, float scale, void* stream) {
    hipStream_t st = (hipStream_t)stream;
    const int nqb = (max_qlen + TPB - 1) / TPB;
    const float sc = scale * QCORR;              // K 侧 e4m3 读成的值小 2^8，这里乘回（与 pa_lib.cpp 同）
    hipLaunchKernelGGL(pp_v1, dim3(nqb, bs), dim3(256), 0, st,
        (u16*)out, (const u16*)q, (const u8*)kc, (const u8*)vc,
        (const int*)bt, (const int*)cu_q, (const int*)slen, n_per, sc);
}
int pp_qfmt() { return QFMT; }
}
