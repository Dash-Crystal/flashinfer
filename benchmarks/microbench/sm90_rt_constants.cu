// Round-trip constants for the sm90 XQA consumer chain (plan P0.3 (d)).
//   RT_mbarrier : 128-thread mbarrier arrive(release.cta) + try_wait(acquire.cta, hint
//                 0xFFFFFFFF) poll loop  -- the kernel's CtaBarrier::arrive_and_wait
//   RT_bar.sync : bar.sync <id>, 128 (NamedBarrier)
//   L_mma       : wgmma.fence; k x m64n8k16 SS bf16 (SW128 descriptors, K-tile-like
//                 layout, distinct k-slices); commit_group; wait_group 0  (k = 1, 4, 8)
// One 128-thread CTA per measurement; one templated kernel per test so the timed
// loop holds nothing but the primitive.  Optional extra CTAs (arg 3) spin on
// try_wait of a never-completing phase on the same SM to mimic waiting warps.
// Build: nvcc -gencode arch=compute_90a,code=sm_90a -O3 -o sm90_rt_constants sm90_rt_constants.cu
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t smem_u32(void const* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ uint64_t mbar_arrive(uint64_t* bar) {
  uint64_t tok;
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];\n" : "=l"(tok) : "r"(smem_u32(bar)) : "memory");
  return tok;
}
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, uint64_t tok) {
  uint32_t ok;
  asm volatile(
      "{\n .reg .pred p;\n"
      " mbarrier.try_wait.acquire.cta.shared::cta.b64 p, [%1], %2, %3;\n"
      " selp.u32 %0, 1, 0, p;\n}\n"
      : "=r"(ok) : "r"(smem_u32(bar)), "l"(tok), "r"(0xFFFFFFFFu) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_arrive_and_wait(uint64_t* bar) {
  uint64_t const tok = mbar_arrive(bar);
  while (!mbar_try_wait(bar, tok)) {
  }
}
__device__ __forceinline__ void bar_sync_128(uint32_t id) {
  asm volatile("bar.sync %0, 128;\n" ::"r"(id) : "memory");
}
// SW128 K-major descriptor as gmma::makeMatDesc(data, 0, rowBytes*8 = 1024, k128) on a
// 1024-aligned buffer: addr>>4 | dimK 0 | dimMN (1024>>4)<<32 | baseOffset 0 | swizzle 1<<62.
__device__ __forceinline__ uint64_t make_desc_sw128(void const* p) {
  uint64_t const addr = (smem_u32(p) & 0x3FFFFu) >> 4;
  uint64_t const dimMN = 1024u >> 4;
  return addr | (dimMN << 32) | (uint64_t(1) << 62);
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_m64n8k16(float (&d)[4], uint64_t descA, uint64_t descB) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "l"(descA), "l"(descB));
}

struct __align__(1024) Smem {
  __align__(1024) uint8_t kTile[8192];  // 64 rows x 128 B (one K part, SW128)
  __align__(1024) uint8_t qTile[1024];  // 8 rows x 128 B
  __align__(8) uint64_t bar[4];
  float colMax[8];
};

enum Test : int {
  kMbarArriveWait = 0, kBarSync, kSyncwarp, kMbarArriveOnly,
  kWgmma1, kWgmma4, kWgmma8, kAtomicMax, kEmpty, kNbTests
};
static char const* kNames[kNbTests] = {
    "mbarrier_arrive_and_wait_128", "bar.sync_128", "syncwarp", "mbarrier_arrive_only_128",
    "wgmma_1x_m64n8k16_fence_commit_wait", "wgmma_4x_m64n8k16_fence_commit_wait",
    "wgmma_8x_m64n8k16_fence_commit_wait", "atomicMax_smem_lane<4_x2", "empty_loop"};

template <int TEST>
__device__ __forceinline__ void body(Smem& s, float (&acc)[4], uint64_t descA, uint64_t descB,
                                     uint32_t tid, int i) {
  if constexpr (TEST == kMbarArriveWait) {
    mbar_arrive_and_wait(&s.bar[0]);
  } else if constexpr (TEST == kBarSync) {
    bar_sync_128(1);
  } else if constexpr (TEST == kSyncwarp) {
    __syncwarp();
  } else if constexpr (TEST == kMbarArriveOnly) {
    (void)mbar_arrive(&s.bar[1]);  // 128 arrivals complete a phase each iteration; nobody waits
  } else if constexpr (TEST == kWgmma1 || TEST == kWgmma4 || TEST == kWgmma8) {
    constexpr int k = TEST == kWgmma1 ? 1 : TEST == kWgmma4 ? 4 : 8;
    wgmma_fence();
#pragma unroll
    for (int j = 0; j < k; j++) wgmma_m64n8k16(acc, descA + 2 * j, descB + 2 * j);
    wgmma_commit();
    wgmma_wait0();
  } else if constexpr (TEST == kAtomicMax) {
    if ((tid & 31) < 4) {
      atomicMax(reinterpret_cast<int*>(&s.colMax[tid & 3]), i);
      atomicMax(reinterpret_cast<int*>(&s.colMax[4 + (tid & 3)]), i + 1);
    }
  } else {
    asm volatile("" ::: "memory");
  }
}

template <int TEST>
__global__ void __launch_bounds__(128) rt_kernel(long long* out, float* sink, int iters, int* smCount) {
  extern __shared__ __align__(1024) uint8_t raw[];
  Smem& s = *reinterpret_cast<Smem*>(raw);
  __shared__ int role;
  uint32_t const tid = threadIdx.x;
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;\n" : "=r"(smid));
  if (tid == 0) {
    for (int i = 0; i < 4; i++) mbar_init(&s.bar[i], 128);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    role = atomicAdd(&smCount[smid], 1);  // first CTA on this SM measures, the rest spin
  }
  for (uint32_t i = tid; i < sizeof(s.kTile) / 4; i += 128) reinterpret_cast<uint32_t*>(s.kTile)[i] = 0;
  for (uint32_t i = tid; i < sizeof(s.qTile) / 4; i += 128) reinterpret_cast<uint32_t*>(s.qTile)[i] = 0;
  if (tid < 8) s.colMax[tid] = -1e30f;
  __syncthreads();
  if (role != 0) {
    // spinner: poll try_wait on a phase that never completes (like a waiting consumer group)
    uint64_t const tok = mbar_arrive(&s.bar[3]);
    while (!mbar_try_wait(&s.bar[3], tok)) {
    }
    uint64_t const tok2 = mbar_arrive(&s.bar[3]);  // second phase: only 128 of 256 needed? no: count 128 -> completes.
    // use bar[2] with a single arrival so the phase never completes
    uint64_t const tok3 = tid == 0 ? mbar_arrive(&s.bar[2]) : 0;
    (void)tok2;
    long long const t0 = clock64();
    while (clock64() - t0 < 6000000LL) {  // ~3 ms, longer than the measuring CTA's loop
      if (mbar_try_wait(&s.bar[2], tok3)) break;
    }
    return;
  }
  uint64_t const descA = make_desc_sw128(s.kTile);
  uint64_t const descB = make_desc_sw128(s.qTile);
  float acc[4] = {0, 0, 0, 0};
  for (int w = 0; w < 16; w++) body<TEST>(s, acc, descA, descB, tid, w);
  __syncthreads();
  long long const t0 = clock64();
  for (int i = 0; i < iters; i++) body<TEST>(s, acc, descA, descB, tid, i);
  long long const t1 = clock64();
  __syncthreads();
  if (tid == 0) out[smid * kNbTests + TEST] = t1 - t0;
  sink[blockIdx.x * 128 + tid] = acc[0] + acc[1] + acc[2] + acc[3];
}

template <int TEST>
static void run(long long* d, float* sink, int iters, int* smCount, int total, size_t smem) {
  cudaFuncSetAttribute(rt_kernel<TEST>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaMemset(smCount, 0, sizeof(int) * 256);
  rt_kernel<TEST><<<total, 128, smem>>>(d, sink, iters, smCount);
}

int main(int argc, char** argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 4000;
  int perSm = argc > 2 ? atoi(argv[2]) : 1;  // CTAs per SM: 1 measuring + (perSm-1) spinning warp groups
  int dev = 0; int clockKHz = 0; cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, dev);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  int const nbSm = prop.multiProcessorCount;
  int total = nbSm * perSm;
  long long* d; float* sink; int* smCount;
  cudaMalloc(&d, sizeof(long long) * kNbTests * 256);
  cudaMalloc(&sink, sizeof(float) * 128 * total);
  cudaMalloc(&smCount, sizeof(int) * 256);
  cudaMemset(d, 0, sizeof(long long) * kNbTests * 256);
  size_t smem = sizeof(Smem) + 1024;
  int ctas = nbSm;
  printf("device %s clock %d kHz smem/CTA %zu B SMs %d CTAs/SM %d (1 measuring + %d spinning) iters %d\n", prop.name, clockKHz, smem, nbSm, perSm, perSm - 1, iters);
  for (int rep = 0; rep < 3; rep++) {
    run<kMbarArriveWait>(d, sink, iters, smCount, total, smem);
    run<kBarSync>(d, sink, iters, smCount, total, smem);
    run<kSyncwarp>(d, sink, iters, smCount, total, smem);
    run<kMbarArriveOnly>(d, sink, iters, smCount, total, smem);
    run<kWgmma1>(d, sink, iters, smCount, total, smem);
    run<kWgmma4>(d, sink, iters, smCount, total, smem);
    run<kWgmma8>(d, sink, iters, smCount, total, smem);
    run<kAtomicMax>(d, sink, iters, smCount, total, smem);
    run<kEmpty>(d, sink, iters, smCount, total, smem);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error %s\n", cudaGetErrorString(e)); return 1; }
    long long* h = new long long[kNbTests * 256];
    cudaMemcpy(h, d, sizeof(long long) * kNbTests * 256, cudaMemcpyDeviceToHost);
    for (int test = 0; test < kNbTests; test++) {
      double mn = 1e30, mx = 0, sum = 0; int n = 0;
      for (int b = 0; b < 256; b++) { if (h[b * kNbTests + test] == 0) continue; double v = (double)h[b * kNbTests + test] / iters; mn = mn < v ? mn : v; mx = mx > v ? mx : v; sum += v; n++; }
      printf("rep %d %-40s cyc/iter min %8.1f mean %8.1f max %8.1f over %d SMs (min %.3f us at %.2f GHz)\n", rep, kNames[test], mn, sum / n, mx, n, mn / (clockKHz * 1e-3), clockKHz * 1e-6);
    }
    delete[] h;
  }
  return 0;
}
