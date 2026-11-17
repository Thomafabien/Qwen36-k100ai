// 自写分页 decode 注意力：编译成共享库供 sglang 调用（运行时 nsplit）
// 由 pa_v9.cpp 生成，内核主体未改动；QFMT=4 选 e4m3、=5 选 e5m2。
#include <hip/hip_runtime.h>
#include <cmath>
// ── 从算子工具箱 opkit_dev.hpp 原样取来的三个设备函数（逐字一致，勿改）
using u8 = unsigned char; using u16 = unsigned short;
typedef _Float16 half_t;
typedef half_t half4 __attribute__((ext_vector_type(4)));
typedef float  f32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ u16 f2bf(float f){ unsigned u=__float_as_uint(f); if(!(f==f)) return 0x7fc0; return (u16)((u + 0x7fffu + ((u>>16)&1u))>>16); }
// 超越函数按 GCN 降级方式，常量用十六进制位模式（十进制写 ln2 低位曾致全盘 1-ulp 漂移）
__device__ __forceinline__ float fexp(float x){ return __builtin_amdgcn_exp2f(x * __uint_as_float(0x3fb8aa3bu)); }
__device__ __forceinline__ unsigned shfl_lane(unsigned v, int src){ return __builtin_amdgcn_ds_bpermute(src << 2, v); }

#define HD   256
#define PAGE 64
#define HQ   4
#ifndef PA_SQ
#define PA_SQ 4             // 4=验证步（q=4）；1=草稿步（q=1）
#endif
#define SQ   PA_SQ
#define NQ   (HQ * SQ)      // 验证步 16 个 query 向量；草稿步 4 个
#define NQMAX 16            // LDS 数组按最大值开，L 恒可到 15
#define NW   4              // 每 block 4 个 wave
#ifndef QFMT
#define QFMT 5              // 量化格式：5=e5m2（与厂商一致）4=e4m3（精度 2 倍，厂商内核不支持）
#endif
#ifndef ABLATE
#define ABLATE 0            // 消融开关：0=全部 1=只读不算 2=读+QK 3=读+QK+softmax 4=全部但去掉P重排的LDS
#endif
#define PA_MAX_SPLIT 512    // combine 的 s_sc 按此开，实际 nsplit 运行时传入（≤ 此值）

__device__ __forceinline__ float bf2fl(u16 h) { return __uint_as_float((unsigned)h << 16); }
// 4 个 fp8 字节 → 4 个 fp16
#if QFMT == 5
// e5m2：与 fp16 指数偏置同为 15、位布局对齐 ⇒ 左移 8 位即得，值精确无需修正
//（256 个位模式已逐位验证，含次正规数与 NaN）
__device__ __forceinline__ half4 e5m2x4(unsigned w) {
    half4 r;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        r[i] = __builtin_bit_cast(half_t, (u16)((u16)((w >> (8 * i)) & 0xFF) << 8));
    return r;
}
#define QDEQ(w) e5m2x4(w)
#define QCORR   1.0f           // 无需修正
#else
// e4m3：偏置 7，与 fp16 的 15 不同 ⇒ 不能直接移位。
// 把 [S][EEEE][MMM] 的符号搬到 bit15、[EEEE][MMM] 左移 7 位当 fp16 读，值统一差 2^8：
//   规格数  ：2^(E-7)·1.M  读成 2^(E-15)·1.M            ⇒ 差 2^8
//   次规格数：M·2^-9       读成 (M·128/1024)·2^-14=M·2^-17 ⇒ 也差 2^8  ← 两种情形同一因子
// 这个 2^8 由 K 侧的 softmax scale 与 V 侧的最终归一化各吸收一次，**零指令开销**。
__device__ __forceinline__ half4 e4m3x4(unsigned w) {
    // 两两一组打包处理：把 4 个字节分成偶/奇两组，每组的两个字节在一个 u32 的两个 u16 里同时算，
    // 省掉逐字节的提取与合并（每组 4 条位运算 + 1 条或，共约 12 条/dword）
    const unsigned ev = w & 0x00FF00FFu;            // 字节 0、2
    const unsigned od = (w >> 8) & 0x00FF00FFu;     // 字节 1、3
    const unsigned fe = ((ev & 0x00800080u) << 8) | ((ev & 0x007F007Fu) << 7);   // fp16(b0) | fp16(b2)<<16
    const unsigned fo = ((od & 0x00800080u) << 8) | ((od & 0x007F007Fu) << 7);   // fp16(b1) | fp16(b3)<<16
    half4 r;
    r[0] = __builtin_bit_cast(half_t, (u16)(fe & 0xFFFFu));
    r[1] = __builtin_bit_cast(half_t, (u16)(fo & 0xFFFFu));
    r[2] = __builtin_bit_cast(half_t, (u16)(fe >> 16));
    r[3] = __builtin_bit_cast(half_t, (u16)(fo >> 16));
    return r;
}
#define QDEQ(w) e4m3x4(w)
#define QCORR   256.0f         // 2^8，见上
#endif

extern "C" __global__ __launch_bounds__(256)
void pa_v3(float* __restrict__ o_part,   // [bs, nsplit, NQ, HD] f32 部分输出
           float* __restrict__ ml,       // [bs, nsplit, NQ, 2]  各 split 的 m / l
           const u16* __restrict__ q,
           const u8* __restrict__ kc, const u8* __restrict__ vc,
           const int* __restrict__ bt, const int* __restrict__ slen,
           int n_per, float scale, int nsplit) {
    const int b = blockIdx.x;
    const int sp = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 63, w = tid >> 6;      // wave 内 lane、第几个 wave
    const int L = lane & 15, H = lane >> 4;       // 片段布局的行列
    const int seqlen = slen[b];
    const int npages = (seqlen + PAGE - 1) / PAGE;

    // ── Q 常驻寄存器：lane(L,H) 持有 Q[query=L][dim=4H+i+16kk]，kk=0..15
    // q 的第 L 个 query 向量 = (token L/HQ, 头 L%HQ)
    half4 qreg[16];
    {
        // NQ<16 时（草稿步 q=1），L>=NQ 的 lane 没有对应 query，Q 填 0、输出丢弃
        const bool qvalid = (L < NQ);
        const u16* qp = q + (qvalid ? ((size_t)b * SQ * HQ + (size_t)(L / HQ) * HQ + (L % HQ)) * HD : 0);
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            half4 t;
#pragma unroll
            for (int i = 0; i < 4; ++i) t[i] = qvalid ? (half_t)bf2fl(qp[kk * 16 + 4 * H + i]) : (half_t)0.0f;
            qreg[kk] = t;
        }
    }

    // ── 在线 softmax 状态与输出累加（每 wave 独立）
    float m_run = -INFINITY, l_run = 0.f;
    f32x4 acc[16];                                  // acc[jj][j] = O[query=L][dim=H+4j+16jj]
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) acc[jj] = (f32x4){0.f, 0.f, 0.f, 0.f};

    __shared__ float s_p[NW][16][17];               // P 的跨 lane 重排（C 布局 → A 布局）
    __shared__ float s_red[NW][NQMAX][2];              // 各 wave 的 m/l，末尾合并用
    __shared__ u8 s_vt[HD][68];                     // 整页 V 转置中转（共享，[256 dim][64 token]+补白）
    __shared__ u8 s_kt[NW][16][68];                 // K 的合并读中转（每 wave 私有，stride 68B 避 bank 冲突）
    // s_out 复用 s_vt 的空间：走到合并阶段时 V 已用完（17408 B ≥ 4×16×64×4=16384 B）
    float (*s_out)[NQMAX][64] = (float (*)[NQMAX][64])&s_vt[0][0];

    // ── 主循环：本 split 负责 [p_lo, p_hi)，wave w 在其中取 w, w+NW, …
    const int per = (npages + nsplit - 1) / nsplit;
    const int p_lo = sp * per, p_hi = min(npages, p_lo + per);
    for (int p = p_lo; p < p_hi; ++p) {          // 4 个 wave 一起处理同一 page（为共享 V）
        const int pg = bt[b * n_per + p];
        const int ntok = min(PAGE, seqlen - p * PAGE);
        const u8* kp = kc + (size_t)pg * PAGE * HD;        // K[token][dim]
        const u8* vp = vc + (size_t)pg * HD * PAGE;        // V[dim][token]

        // 协作把整页 V 合并读进 LDS：256 线程，线程 t 读 dim t 的全部 64 个 token（连续 64 B）
        // bank 检查：写地址 dword = t*17 + c，17 与 32 互质 ⇒ 同周期 32 个线程落在 32 个不同 bank
#if ABLATE != 5
        __syncthreads();
#endif
#pragma unroll
        for (int c = 0; c < 4; ++c)
            *(u32x4*)&s_vt[tid][c * 16] = *(const u32x4*)(vp + (size_t)tid * PAGE + c * 16);
#if ABLATE != 5
        __syncthreads();
#endif

#if ABLATE == 1 || ABLATE == 5
        acc[0][0] += (float)s_vt[tid][0];      // 只做 V 页填充，消费一下防优化
        continue;
#endif
        {                                              // wave w 负责本页的第 w 个 16-token 块
            const int t0 = 16 * w;
            if (t0 < ntok) {
            // ① S = Q·Kᵀ：分 4 轮，每轮先把 16 token × 64 dim 合并读进 LDS，再发 4 条 mmac
            f32x4 s = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int r4 = 0; r4 < 4; ++r4) {
                const int d0 = r4 * 64;
                // 合并读：lane(L,H) 取 token t0+4rep+H 的 dim d0+4L..+3；
                // 16 个连续 lane（L=0..15，H 固定）地址差 4 字节 ⇒ 一次 64 字节连续请求
#pragma unroll
                for (int rep = 0; rep < 4; ++rep) {
                    const unsigned v = *(const unsigned*)(kp + (size_t)(t0 + 4 * rep + H) * HD + d0 + 4 * L);
                    *(unsigned*)&s_kt[w][4 * rep + H][4 * L] = v;
                }
                // 只等 LDS（lgkmcnt(0) = 0xC07F，编码取自厂商反汇编 BF8CC07F）；
                // 不能用 s_waitcnt(0)，那会连 global load 一起等，把访存流水串行化（实测因此只剩 0.8% 收益）
                __builtin_amdgcn_s_waitcnt(0xC07F);
                // 取回 mmac 要的 B[4H+i][L] = K[token=t0+L][dim=d0+16kk+4H+i]
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const unsigned raw = *(const unsigned*)&s_kt[w][L][16 * kk + 4 * H];
#if ABLATE == 1
                    s[0] += (float)(raw & 0xFF);        // 不发 mmac，只把数据消费掉
#else
                    s = __builtin_amdgcn_mmac_f32_16x16x16f16(qreg[r4 * 4 + kk], QDEQ(raw), s);
#endif
                }
            }
            // s[j] = S[query=L][token=t0+H+4j]
#if ABLATE == 1
            // 只读不算：把读到的值累进 acc 防止被优化掉
            acc[0][0] += (float)s[0] + (float)s[1];
            continue;
#endif
#if ABLATE == 2
            acc[0][0] += s[0] + s[1] + s[2] + s[3];
            continue;
#endif
            // ② 在线 softmax：先求本块内 row L 的最大值（跨 j 与跨 H）
            float mb = -INFINITY;
            // 因果掩码：本 lane 的 query 是序列末尾第 (L/HQ) 个，绝对位置 seqlen-SQ+(L/HQ)；
            // 只能看到绝对位置 ≤ 它的 KV。缺这一条时第 0/1/2 个 query 会各多看 3/2/1 个 token
            // ——随机数据下被 softmax 摊薄到看不出来（曾误判为对拍通过），真实数据会暴露。
            const int qpos = seqlen - SQ + (L / HQ);      // SQ=1 时即 seqlen-1（唯一 query 在末尾）
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int tok = t0 + H + 4 * j;
                const int gtok = p * PAGE + tok;          // 该 KV 在序列中的绝对位置
                float v = (tok < ntok && gtok <= qpos) ? s[j] * scale : -INFINITY;
                s[j] = v; mb = fmaxf(mb, v);
            }
#pragma unroll
            for (int d = 16; d <= 32; d <<= 1)              // 跨 H 蝶形归约：lane^16、lane^32 两步
                mb = fmaxf(mb, __uint_as_float(shfl_lane(__float_as_uint(mb), lane ^ d)));
            const float m_new = fmaxf(m_run, mb);
            const float corr = (m_run == -INFINITY) ? 0.f : fexp(m_run - m_new);
            float lb = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) { s[j] = (s[j] == -INFINITY) ? 0.f : fexp(s[j] - m_new); lb += s[j]; }
#pragma unroll
            for (int d = 16; d <= 32; d <<= 1)              // 同上；三步写法会把和累加两遍（曾致输出恰好减半）
                lb += __uint_as_float(shfl_lane(__float_as_uint(lb), lane ^ d));
            l_run = l_run * corr + lb;
            m_run = m_new;

#if ABLATE == 3
            acc[0][0] += s[0] + l_run;
            continue;
#endif
            // ③ P 重排：C 布局 P[L][H+4j] → A 布局 P[L][4H+i]（走 LDS，16×17 避 bank 冲突）
            half4 pa;
#if ABLATE == 4
#pragma unroll
            for (int i = 0; i < 4; ++i) pa[i] = (half_t)s[i];     // 跳过 LDS 往返（数值错，仅计时）
#else
#pragma unroll
            for (int j = 0; j < 4; ++j) s_p[w][L][H + 4 * j] = s[j];
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();
#pragma unroll
            for (int i = 0; i < 4; ++i) pa[i] = (half_t)s_p[w][L][4 * H + i];
#endif

            // ④ O += P·V：B[4H+i][L] = V[token=t0+4H+i][dim=L+16jj]
            //    V 转置存 → vp[dim][token]，lane 读 4 个连续 token
#pragma unroll
            for (int jj = 0; jj < 16; ++jj) {
                // B[4H+i][L] = V[token=t0+4H+i][dim=16jj+L]；LDS 里是 s_vt[dim][token]
                // 读地址 dword = (16jj+L)*17 + (t0+4H)/4，L 跨 17 dword ⇒ 无 bank 冲突
                const unsigned raw = *(const unsigned*)&s_vt[16 * jj + L][t0 + 4 * H];
                f32x4 o = acc[jj];
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] *= corr;
                acc[jj] = __builtin_amdgcn_mmac_f32_16x16x16f16(pa, QDEQ(raw), o);
            }
            }
        }
    }

    // ── block 内 4 个 wave 合并 → 写本 split 的部分结果（不归一化，留给 combine）
    if (L < NQ) { s_red[w][L][0] = m_run; s_red[w][L][1] = l_run; }   // L>=NQ 的 lane 不写
    __syncthreads();
    float mg = -INFINITY, lg = 0.f, sc[NW];
#pragma unroll
    for (int k = 0; k < NW; ++k) mg = fmaxf(mg, s_red[k][L][0]);
#pragma unroll
    for (int k = 0; k < NW; ++k) {
        sc[k] = (s_red[k][L][0] == -INFINITY) ? 0.f : fexp(s_red[k][L][0] - mg);
        lg += s_red[k][L][1] * sc[k];
    }
    // ⚠ 必须带 L < NQ：NQ<16 时（草稿步 q=1，NQ=4）每个 split 只有 NQ*2 个 float，
    // 而 L 恒可到 15，漏掉这个守卫会越界覆盖相邻 split 的 m/l——
    // 单 split 时越界写到不存在的下一个 split，combine 读不到，症状被完全掩盖。
    if (w == 0 && H == 0 && L < NQ) {
        ml[((size_t)b * nsplit + sp) * NQ * 2 + L * 2 + 0] = mg;
        ml[((size_t)b * nsplit + sp) * NQ * 2 + L * 2 + 1] = lg;
    }
    float* op = o_part + ((size_t)b * nsplit + sp) * NQ * HD + (size_t)(L < NQ ? L : 0) * HD;
    for (int batch = 0; batch < 4; ++batch) {
        __syncthreads();
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int jj = batch * 4 + t;
#pragma unroll
            for (int j = 0; j < 4; ++j) s_out[w][L][H + 4 * j + 16 * t] = acc[jj][j];
        }
        __syncthreads();
        if (w == 0 && L < NQ) {
            for (int d = H; d < 64; d += 4) {
                float v = 0.f;
#pragma unroll
                for (int k = 0; k < NW; ++k) v += s_out[k][L][d] * sc[k];
                op[batch * 64 + d] = v * QCORR;
            }
        }
    }
}

// ── 合并各 split：grid=(bs, NQ)，每 block 256 线程分摊 HD
// v5b 修正：原先由 tid==0 串行读 nsplit 个 m/l（每次都是未缓存 global，nsplit 越大越慢，
// 正好抵消主内核提高 nsplit 的并行度收益）；改为 256 线程并行读 + block 内归约。
extern "C" __global__ __launch_bounds__(256)
void pa_combine(u16* __restrict__ out, const float* __restrict__ o_part,
                const float* __restrict__ ml, int nsplit_used, int nsplit) {
    const int b = blockIdx.x, qi = blockIdx.y;
    const int tid = threadIdx.x;
    __shared__ float s_sc[PA_MAX_SPLIT];
    __shared__ float s_red2[256];
    const float* mlp = ml + (size_t)b * nsplit * NQ * 2 + qi * 2;

    float my_m = -INFINITY;
    for (int k = tid; k < nsplit_used; k += 256) my_m = fmaxf(my_m, mlp[(size_t)k * NQ * 2]);
    s_red2[tid] = my_m; __syncthreads();
    for (int t = 128; t > 0; t >>= 1) { if (tid < t) s_red2[tid] = fmaxf(s_red2[tid], s_red2[tid + t]); __syncthreads(); }
    const float mg = s_red2[0];
    __syncthreads();

    float my_l = 0.f;
    for (int k = tid; k < nsplit_used; k += 256) {
        const float mk = mlp[(size_t)k * NQ * 2], lk = mlp[(size_t)k * NQ * 2 + 1];
        const float c = (mk == -INFINITY) ? 0.f : fexp(mk - mg);
        s_sc[k] = c; my_l += lk * c;
    }
    s_red2[tid] = my_l; __syncthreads();
    for (int t = 128; t > 0; t >>= 1) { if (tid < t) s_red2[tid] += s_red2[tid + t]; __syncthreads(); }
    const float inv = (s_red2[0] > 0.f) ? 1.f / s_red2[0] : 0.f;
    __syncthreads();

    const int tok = qi / HQ, hq = qi % HQ;
    u16* op = out + ((size_t)b * SQ * HQ + (size_t)tok * HQ + hq) * HD;
    const float* opart = o_part + (size_t)b * nsplit * NQ * HD + (size_t)qi * HD;
    for (int d = tid; d < HD; d += 256) {
        float v = 0.f;
        for (int k = 0; k < nsplit_used; ++k) v += opart[(size_t)k * NQ * HD + d] * s_sc[k];
        op[d] = f2bf(v * inv);
    }
}


// ── 对外 C 接口 ───────────────────────────────────────────────────────
// 形状固定为生产的验证步：q 头 4 / kv 头 1 / head_dim 256 / page 64 / seqlen_q 4
// o_part、ml 由调用方预分配（大小见下），以便在 CUDA graph 内复用同一块显存。
extern "C" {

int pa_workspace_bytes(int bs, int nsplit, int* o_part_bytes, int* ml_bytes) {
    const size_t op = (size_t)bs * nsplit * NQ * HD * sizeof(float);
    const size_t ml = (size_t)bs * nsplit * NQ * 2 * sizeof(float);
    if (o_part_bytes) *o_part_bytes = (int)op;
    if (ml_bytes)     *ml_bytes     = (int)ml;
    return (int)(op + ml);
}

// 按总页数挑 nsplit。这是**拟合四个实测点**的查找式启发式，不是推导出来的：
//   总页数(=bs×npages)  最优 block 数(=bs×ns)   实测
//   128   (8192×1)      64     ns64  44.0 µs（ns32 51.1 / ns128 61.4）
//   2048  (32768×4)     128    ns32 169.6 µs（ns64 222.3 / ns128 234.2）
//   2048  (131072×1)    128    ns128 178.9 µs（ns64 201.6 / ns256 215.6）
//   8192  (131072×4)    1024   ns256 719.7 µs（ns128 798.3 / ns512 781.6）
// 注意同为 2048 总页时，bs=1 与 bs=4 的最优 per 不同（16 vs 8），机制未查清（疑似 L2 竞争），
// 所以这里按总页数分档而非按 per。可用环境变量 SGLANG_PA_NSPLIT 覆盖。
int pa_pick_nsplit(int npages, int bs) {
    const long total = (long)npages * bs;
    int target_blocks = (total <= 256) ? 64 : (total <= 4096) ? 128 : 1024;
    int ns = target_blocks / (bs > 0 ? bs : 1);
    if (ns < 8) ns = 8;
    int p = 8; while (p < ns && p < PA_MAX_SPLIT) p <<= 1;
    return p;
}

void pa_decode(void* out, const void* q, const void* kc, const void* vc,
               const int* bt, const int* slen, float* o_part, float* ml,
               int bs, int n_per, int npages, int nsplit, float scale_in, void* stream) {
    const float scale = scale_in * QCORR;      // 吸收 e4m3 的 2^8（e5m2 时 QCORR=1）
    const int per = (npages + nsplit - 1) / nsplit;
    const int nsplit_used = (npages + per - 1) / per;
    hipStream_t st = (hipStream_t)stream;
    hipLaunchKernelGGL(pa_v3, dim3(bs, nsplit), dim3(256), 0, st,
                       o_part, ml, (const u16*)q, (const u8*)kc, (const u8*)vc,
                       bt, slen, n_per, scale, nsplit);
    hipLaunchKernelGGL(pa_combine, dim3(bs, NQ), dim3(256), 0, st,
                       (u16*)out, o_part, ml, nsplit_used, nsplit);
}

int pa_qfmt() { return QFMT; }
int pa_sq()   { return SQ; }

}   // extern "C"
