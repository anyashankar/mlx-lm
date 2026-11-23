// GPU memory-consistency litmus tests across thread blocks (CUDA).
//
// Each test instance is two threads in DIFFERENT blocks (the analogue of two
// Metal threadgroups), touching two locations x and y on different cache
// lines. Instances are scattered through memory with a fresh random
// permutation every iteration, and extra "stress" blocks hammer an unrelated
// buffer to perturb timing, as in GPU litmus testing practice.
//
// Tests (initially x = y = 0; r0, r1 are loaded values):
//   MP  message passing   T0: x=1; y=1          T1: r0=y; r1=x     weak: r0=1, r1=0
//   SB  store buffering   T0: x=1; r0=y         T1: y=1; r1=x      weak: r0=0, r1=0
//   LB  load buffering    T0: r0=x; y=1         T1: r1=y; x=1      weak: r0=1, r1=1
//
// Variants (what sits between each thread's two operations):
//   relaxed       device-scope atomics, memory_order_relaxed, no fence
//   plain         ordinary (non-atomic) loads/stores; the reader first loads
//                 the data location so a stale copy can sit in L1 (Volta+
//                 caches ordinary global loads in L1, which is not coherent)
//   fence_block   relaxed atomics + atomic_thread_fence(seq_cst, block scope)
//                 (too narrow a scope: the other thread is in another block)
//   fence_device  relaxed atomics + atomic_thread_fence(seq_cst, device scope)
//   ordered       MP/LB: release stores + acquire loads; SB: seq_cst ops
// Compiler barriers sit between the two operations in every variant, so any
// weak outcome comes from the hardware, not from compiler reordering.
//
// The PTX memory model forbids the weak outcome for fence_device and ordered;
// observing one there is a model violation and is flagged. For relaxed,
// plain and fence_block it is allowed (it may or may not show up).
//
// Build: nvcc -O2 -std=c++17 -arch=native litmus.cu -o litmus
// Run:   ./litmus [--iters 2000] [--inst 1024] [--stress 128] [--csv litmus.csv]

#include <cuda/atomic>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                                  \
  do {                                                                                         \
    cudaError_t e = (x);                                                                       \
    if (e != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
      std::exit(2);                                                                            \
    }                                                                                          \
  } while (0)

enum Test { MP = 0, SB = 1, LB = 2 };
enum Variant { RELAXED = 0, PLAIN = 1, FENCE_BLOCK = 2, FENCE_DEVICE = 3, ORDERED = 4 };
static const char* TEST_NAMES[] = {"MP", "SB", "LB"};
static const char* VAR_NAMES[] = {"relaxed", "plain", "fence_block", "fence_device", "ordered"};

using Ref = cuda::atomic_ref<int, cuda::thread_scope_device>;
constexpr int STRIDE = 64;            // ints between instances (256 B)
constexpr int Y_OFF = 32;             // y is 128 B after x: a different cache line
constexpr long SPIN_LIMIT = 1L << 22; // soft-barrier timeout

__device__ __forceinline__ void cbar() { asm volatile("" ::: "memory"); }

template <int V>
__device__ __forceinline__ void mid_fence() {
  cbar();
  if (V == FENCE_BLOCK) cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block);
  if (V == FENCE_DEVICE) cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
  cbar();
}

template <int T, int V>
__device__ __forceinline__ void st(int* p, bool second) {
  if (V == PLAIN) { *p = 1; return; }
  cuda::memory_order o = cuda::memory_order_relaxed;
  if (V == ORDERED) o = (T == SB) ? cuda::memory_order_seq_cst : (second ? cuda::memory_order_release : cuda::memory_order_relaxed);
  Ref(*p).store(1, o);
}

template <int T, int V>
__device__ __forceinline__ int ld(int* p, bool first) {
  if (V == PLAIN) { cbar(); int v = *p; cbar(); return v; }
  cuda::memory_order o = cuda::memory_order_relaxed;
  if (V == ORDERED) o = (T == SB) ? cuda::memory_order_seq_cst : (first ? cuda::memory_order_acquire : cuda::memory_order_relaxed);
  return Ref(*p).load(o);
}

template <int T, int V>
__global__ void litmus(int* loc, const int* perm, int* sync, int* res, int* invalid, int n, int* stress,
                       int stress_words, unsigned seed) {
  int b = blockIdx.x;
  if (b >= 2 * n) {  // stress block: scattered read-modify-writes on its own buffer
    unsigned h = seed ^ (b * 0x9E3779B9u) ^ (threadIdx.x * 0x85EBCA6Bu);
    for (int i = 0; i < 256; i++) {
      h ^= h << 13; h ^= h >> 17; h ^= h << 5;
      int w = (int)(h % (unsigned)stress_words);
      stress[w] = stress[w] + 1;
    }
    return;
  }
  if (threadIdx.x != 0) return;
  int inst = b % n, role = b / n;  // role-0 blocks come first, role-1 blocks n later
  int* x = loc + (size_t)perm[inst] * STRIDE;
  int* y = x + Y_OFF;

  // PLAIN: the reader warms its L1 with the initial value of what it reads last.
  int warm = 0;
  if (V == PLAIN) {
    if (T == MP && role == 1) warm = *x;
    if (T == SB) warm = role == 0 ? *y : *x;
    cbar();
  }

  // Soft barrier so both threads start together. Bounded: blocks are not
  // guaranteed to be co-resident, so never wait forever on the partner.
  Ref s(sync[inst]);
  s.fetch_add(1, cuda::memory_order_relaxed);
  long spins = 0;
  while (s.load(cuda::memory_order_relaxed) < 2 && ++spins < SPIN_LIMIT) {
  }
  if (spins >= SPIN_LIMIT) {
    invalid[inst] = 1;
    return;
  }

  auto load = [&](int* p, bool first) -> int { return ld<T, V>(p, first); };

  if (T == MP) {
    if (role == 0) { st<T, V>(x, false); mid_fence<V>(); st<T, V>(y, true); }
    else { int r0 = load(y, true); mid_fence<V>(); int r1 = load(x, false); res[2 * inst] = r0; res[2 * inst + 1] = r1; }
  } else if (T == SB) {
    if (role == 0) { st<T, V>(x, false); mid_fence<V>(); res[2 * inst] = load(y, true); }
    else { st<T, V>(y, false); mid_fence<V>(); res[2 * inst + 1] = load(x, true); }
  } else {  // LB
    if (role == 0) { int r0 = load(x, true); mid_fence<V>(); st<T, V>(y, true); res[2 * inst] = r0; }
    else { int r1 = load(y, true); mid_fence<V>(); st<T, V>(x, true); res[2 * inst + 1] = r1; }
  }
  if (V == PLAIN && warm == 12345) res[0] = warm;  // keep the warm-up load alive
}

typedef void (*Kernel)(int*, const int*, int*, int*, int*, int, int*, int, unsigned);

template <int T>
Kernel pick(int v) {
  switch (v) {
    case RELAXED: return litmus<T, RELAXED>;
    case PLAIN: return litmus<T, PLAIN>;
    case FENCE_BLOCK: return litmus<T, FENCE_BLOCK>;
    case FENCE_DEVICE: return litmus<T, FENCE_DEVICE>;
    default: return litmus<T, ORDERED>;
  }
}

Kernel kernel_for(int t, int v) { return t == MP ? pick<MP>(v) : t == SB ? pick<SB>(v) : pick<LB>(v); }

bool model_forbids(int v) { return v == FENCE_DEVICE || v == ORDERED; }

int main(int argc, char** argv) {
  int iters = 2000, n = 1024, stress_blocks = 128;
  std::string csv = "litmus.csv";
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--inst" && i + 1 < argc) n = std::atoi(argv[++i]);
    else if (a == "--stress" && i + 1 < argc) stress_blocks = std::atoi(argv[++i]);
    else if (a == "--csv" && i + 1 < argc) csv = argv[++i];
  }
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s (sm_%d%d, %d SMs); %d instances x %d iterations per test, %d stress blocks\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount, n, iters, stress_blocks);

  const int stress_words = 1 << 20;
  int *loc, *perm_d, *sync, *res, *invalid, *stress;
  CK(cudaMalloc(&loc, (size_t)n * STRIDE * sizeof(int)));
  CK(cudaMalloc(&perm_d, n * sizeof(int)));
  CK(cudaMalloc(&sync, n * sizeof(int)));
  CK(cudaMalloc(&res, 2 * n * sizeof(int)));
  CK(cudaMalloc(&invalid, n * sizeof(int)));
  CK(cudaMalloc(&stress, stress_words * sizeof(int)));
  CK(cudaMemset(stress, 0, stress_words * sizeof(int)));

  std::vector<int> perm(n), h_res(2 * n), h_inv(n);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937 rng(12345);

  FILE* out = std::fopen(csv.c_str(), "w");
  std::fprintf(out, "device,test,variant,valid,invalid,r00,r01,r10,r11,weak,weak_rate,model_forbids_weak\n");
  std::printf("\n%-4s %-13s %10s %8s %10s %10s %10s %10s %10s  %s\n", "test", "variant", "valid", "invalid",
              "(0,0)", "(0,1)", "(1,0)", "(1,1)", "weak", "verdict");
  int violations = 0;
  for (int t = 0; t < 3; t++)
    for (int v = 0; v < 5; v++) {
      long long cnt[4] = {0, 0, 0, 0}, inval = 0, other = 0;
      Kernel k = kernel_for(t, v);
      for (int it = 0; it < iters; it++) {
        std::shuffle(perm.begin(), perm.end(), rng);
        CK(cudaMemcpy(perm_d, perm.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemset(loc, 0, (size_t)n * STRIDE * sizeof(int)));
        CK(cudaMemset(sync, 0, n * sizeof(int)));
        CK(cudaMemset(res, 0xFF, 2 * n * sizeof(int)));
        CK(cudaMemset(invalid, 0, n * sizeof(int)));
        k<<<2 * n + stress_blocks, 64>>>(loc, perm_d, sync, res, invalid, n, stress, stress_words, (unsigned)rng());
        CK(cudaGetLastError());
        CK(cudaMemcpy(h_res.data(), res, 2 * n * sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_inv.data(), invalid, n * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++) {
          int r0 = h_res[2 * i], r1 = h_res[2 * i + 1];
          if (h_inv[i] || r0 < 0 || r1 < 0) { inval++; continue; }
          if (r0 > 1 || r1 > 1) { other++; continue; }
          cnt[r0 * 2 + r1]++;
        }
      }
      long long valid = cnt[0] + cnt[1] + cnt[2] + cnt[3];
      long long weak = t == MP ? cnt[2] : t == SB ? cnt[0] : cnt[3];
      bool forbids = model_forbids(v);
      bool viol = forbids && weak > 0;
      violations += viol;
      const char* verdict = viol ? "VIOLATION (model forbids this)" : forbids ? "ok (forbidden, not seen)"
                                  : weak ? "allowed, observed" : "allowed, not observed";
      std::printf("%-4s %-13s %10lld %8lld %10lld %10lld %10lld %10lld %10lld  %s\n", TEST_NAMES[t], VAR_NAMES[v],
                  valid, inval + other, cnt[0], cnt[1], cnt[2], cnt[3], weak, verdict);
      std::fprintf(out, "%s,%s,%s,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.3e,%d\n", prop.name, TEST_NAMES[t],
                   VAR_NAMES[v], valid, inval + other, cnt[0], cnt[1], cnt[2], cnt[3], weak,
                   valid ? (double)weak / valid : 0.0, forbids);
    }
  std::fclose(out);
  std::printf("\nweak = MP (1,0), SB (0,0), LB (1,1). invalid = partner block never arrived within the spin limit.\n");
  std::printf("results -> %s%s\n", csv.c_str(), violations ? "   ** model violations observed **" : "");
  return violations ? 1 : 0;
}
