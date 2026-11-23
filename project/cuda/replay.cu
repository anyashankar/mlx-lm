// CUDA trace-replay harness for the MLX MoE traces (see TRACE_FORMAT.md).
//
// Loads MLX's quantized expert weights (4- or 8-bit per projection) exactly as MLX stores them, dequantizes on
// the fly, recomputes every recorded expert output
//     y[row, slot] = down( silu(gate x) * (up x) ),  expert = idx[row, slot]
// and compares with the outputs MLX produced on the M5 Max. It also checks
// the dequantizer alone against MLX's dequantization of one expert.
//
// The kernels are a simple, correct reference (one warp per output value),
// not the optimized fused kernel; they establish that loading, layout and
// math match MLX, which the fused kernel is then checked against.
//
// Build: nvcc -O2 -std=c++17 -arch=native replay.cu -o replay   (GPU + CPU paths)
//        g++  -O2 -std=c++17 -x c++ replay.cu -o replay_cpu     (CPU path only)
// Run:   ./replay <trace_dir> [--cpu] [--bf16 | --fp32] [--strict-tol 1e-4] [--tol 0.02] [--cos 0.9999]
//        bf16 rounding is emulated automatically when the trace's dtype is bfloat16.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "npy.h"

#ifdef __CUDACC__
#define HD __host__ __device__
#include <cuda_runtime.h>
#define CK(x)                                                                     \
  do {                                                                            \
    cudaError_t e = (x);                                                          \
    if (e != cudaSuccess) {                                                       \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
      std::exit(2);                                                               \
    }                                                                             \
  } while (0)
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// MLX affine layout (verified against mx.quantize / mx.dequantize, 4 and 8 bits):
//   q[e][o][w]  uint32, per = 32/bits values per word, w = i / per; value i sits in
//               bits bits*(i%per) .. bits*(i%per)+bits-1 (lowest bits first)
//   s[e][o][g], b[e][o][g]  one scale and bias per group of `group` inputs
//   W[o][i] = s[o][i/group] * nibble + b[o][i/group]
// The matmul is y = W x (MLX calls gather_qmm with transpose=True).
// ---------------------------------------------------------------------------
struct QMat {           // one projection, all experts
  const uint32_t* q;    // [E, out, in*bits/32]
  const float* s;       // [E, out, in/group]
  const float* b;       // [E, out, in/group]
  int E, out, in, group, bits;  // bits = 4 or 8 (mlx-lm keeps Granite's down_proj at 8)
};

HD inline float deq_one(const QMat& m, int e, int o, int i) {
  int per = 32 / m.bits;  // values per word: 8 (4-bit) or 4 (8-bit)
  uint32_t mask = (1u << m.bits) - 1u;
  size_t row = (size_t)e * m.out + o;
  uint32_t w = m.q[row * (m.in / per) + i / per];
  float v = (float)((w >> (m.bits * (i % per))) & mask);
  size_t g = row * (m.in / m.group) + i / m.group;
  return m.s[g] * v + m.b[g];
}

// Partial dot product of row o of expert e with x, over packed words
// lane, lane + nlanes, ... (nlanes = 32 on the GPU, 1 on the CPU).
template <int BITS>
HD inline float dot_qb(const QMat& m, int e, int o, const float* x, int lane, int nlanes) {
  constexpr int PER = 32 / BITS;
  constexpr uint32_t MASK = (1u << BITS) - 1u;
  size_t row = (size_t)e * m.out + o;
  const uint32_t* qr = m.q + row * (m.in / PER);
  const float* sr = m.s + row * (m.in / m.group);
  const float* br = m.b + row * (m.in / m.group);
  float acc = 0.f;
  for (int w = lane; w < m.in / PER; w += nlanes) {
    uint32_t v = qr[w];
    int i0 = w * PER;
    float sc = sr[i0 / m.group], bi = br[i0 / m.group];  // group is a multiple of PER
    float part = 0.f, xs = 0.f;
#pragma unroll
    for (int j = 0; j < PER; j++) {
      float xv = x[i0 + j];
      part += (float)((v >> (BITS * j)) & MASK) * xv;
      xs += xv;
    }
    acc += sc * part + bi * xs;  // sum_j (s*q_j + b) x_j
  }
  return acc;
}

HD inline float dot_q(const QMat& m, int e, int o, const float* x, int lane, int nlanes) {
  return m.bits == 8 ? dot_qb<8>(m, e, o, x, lane, nlanes) : dot_qb<4>(m, e, o, x, lane, nlanes);
}

// Round to bfloat16 (round to nearest even), returned as float.
HD inline float bf16r(float v) {
#ifdef __CUDA_ARCH__
  uint32_t u = __float_as_uint(v);
#else
  uint32_t u;
  std::memcpy(&u, &v, 4);
#endif
  u += 0x7FFFu + ((u >> 16) & 1u);
  u &= 0xFFFF0000u;
#ifdef __CUDA_ARCH__
  return __uint_as_float(u);
#else
  std::memcpy(&v, &u, 4);
  return v;
#endif
}

// Expert activation. emu=1 rounds where MLX's bf16 kernels round:
// gather_qmm outputs (gate, up), sigmoid, gate*sigmoid, and *up.
HD inline float swiglu(float g, float u, int emu) {
  if (!emu) return g / (1.f + expf(-g)) * u;
  g = bf16r(g);
  u = bf16r(u);
  float sig = bf16r(1.f / (1.f + expf(-g)));
  return bf16r(bf16r(g * sig) * u);
}

// ---------------------------------------------------------------------------
// CPU reference path (same math, serial)
// ---------------------------------------------------------------------------
void moe_cpu(const QMat& G, const QMat& U, const QMat& D, const float* x, const int* idx,
             int rows, int topk, int emu, float* y) {
  int hidden = G.in, inter = G.out;
  std::vector<float> act(inter);
  for (int r = 0; r < rows; r++)
    for (int k = 0; k < topk; k++) {
      int e = idx[r * topk + k];
      const float* xr = x + (size_t)r * hidden;
      for (int j = 0; j < inter; j++)
        act[j] = swiglu(dot_q(G, e, j, xr, 0, 1), dot_q(U, e, j, xr, 0, 1), emu);
      for (int o = 0; o < hidden; o++) {
        float v = dot_q(D, e, o, act.data(), 0, 1);
        y[((size_t)r * topk + k) * hidden + o] = emu ? bf16r(v) : v;
      }
    }
}

#ifdef __CUDACC__
// ---------------------------------------------------------------------------
// GPU reference kernels: one warp per output value, lanes stride the input
// ---------------------------------------------------------------------------
constexpr int WARPS = 4;

__device__ inline float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// grid (ceil(inter / WARPS), rows * topk)
__global__ void k_gate_up(QMat G, QMat U, const float* x, const int* idx, int topk, int emu, float* act) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int j = blockIdx.x * WARPS + warp;
  int pair = blockIdx.y;
  if (j >= G.out) return;  // whole warp exits together, so the shuffles stay convergent
  int e = idx[pair], row = pair / topk;
  const float* xr = x + (size_t)row * G.in;
  float g = warp_sum(dot_q(G, e, j, xr, lane, 32));
  float u = warp_sum(dot_q(U, e, j, xr, lane, 32));
  if (lane == 0) act[(size_t)pair * G.out + j] = swiglu(g, u, emu);
}

// grid (ceil(hidden / WARPS), rows * topk)
__global__ void k_down(QMat D, const float* act, const int* idx, int emu, float* y) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int o = blockIdx.x * WARPS + warp;
  int pair = blockIdx.y;
  if (o >= D.out) return;
  int e = idx[pair];
  float v = warp_sum(dot_q(D, e, o, act + (size_t)pair * D.in, lane, 32));
  if (lane == 0) y[(size_t)pair * D.out + o] = emu ? bf16r(v) : v;
}

__global__ void k_dequant(QMat m, int e, float* out) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)m.out * m.in) return;
  out[t] = deq_one(m, e, (int)(t / m.in), (int)(t % m.in));
}

template <class T> T* to_dev(const Npy& a) {
  T* d;
  CK(cudaMalloc(&d, a.data.size()));
  CK(cudaMemcpy(d, a.data.data(), a.data.size(), cudaMemcpyHostToDevice));
  return d;
}
#endif

// ---------------------------------------------------------------------------
struct Meta {
  std::map<std::string, std::string> kv;
  int geti(const std::string& k) const {
    auto it = kv.find(k);
    if (it == kv.end()) throw std::runtime_error("meta.txt lacks " + k);
    return std::stoi(it->second);
  }
  std::vector<int> list(const std::string& k) const {
    std::vector<int> v;
    auto it = kv.find(k);
    if (it == kv.end()) return v;
    std::istringstream ss(it->second);
    int x;
    while (ss >> x) v.push_back(x);
    return v;
  }
};

Meta read_meta(const std::string& dir) {
  Meta m;
  std::ifstream f(dir + "/meta.txt");
  if (!f) throw std::runtime_error("cannot open " + dir + "/meta.txt");
  std::string line;
  while (std::getline(f, line)) {
    size_t sp = line.find(' ');
    if (sp == std::string::npos) continue;
    m.kv[line.substr(0, sp)] = line.substr(sp + 1);
  }
  return m;
}

struct Err { double maxabs, refmax, cosine; };

Err compare(const float* a, const float* ref, size_t n) {
  double ma = 0, rm = 0, dot = 0, na = 0, nr = 0;
  for (size_t i = 0; i < n; i++) {
    double d = std::fabs((double)a[i] - ref[i]);
    if (d > ma) ma = d;
    if (std::fabs(ref[i]) > rm) rm = std::fabs(ref[i]);
    dot += (double)a[i] * ref[i];
    na += (double)a[i] * a[i];
    nr += (double)ref[i] * ref[i];
  }
  return {ma, rm, dot / (std::sqrt(na * nr) + 1e-30)};
}

void check_shape(const Npy& a, const char* descr, std::vector<int64_t> shape, const std::string& name) {
  if (a.descr != descr || a.shape != shape) {
    std::string got;
    for (auto s : a.shape) got += std::to_string(s) + " ";
    throw std::runtime_error(name + ": expected dtype " + descr + ", got " + a.descr + " with shape [ " + got + "]");
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <trace_dir> [--cpu] [--tol 0.03] [--cos 0.999]\n", argv[0]);
    return 2;
  }
  std::string dir = argv[1];
  bool cpu = false;
  int emu = -1;  // -1: follow the trace's dtype
  double tol = 0.02, cos_min = 0.9999, strict_tol = 1e-4;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--cpu") cpu = true;
    else if (a == "--fp32") emu = 0;
    else if (a == "--bf16") emu = 1;
    else if (a == "--tol" && i + 1 < argc) tol = std::atof(argv[++i]);
    else if (a == "--cos" && i + 1 < argc) cos_min = std::atof(argv[++i]);
    else if (a == "--strict-tol" && i + 1 < argc) strict_tol = std::atof(argv[++i]);
  }
#ifndef __CUDACC__
  cpu = true;
#endif
  Meta meta = read_meta(dir);
  const int E = meta.geti("num_experts"), topk = meta.geti("top_k");
  const int hidden = meta.geti("hidden"), inter = meta.geti("intermediate");
  const int group = meta.geti("group_size"), ref_e = meta.geti("ref_expert");
  // per-projection bits and group (older traces have one "bits" for all three)
  std::map<std::string, int> pbits, pgroup;
  for (const char* n : {"gate", "up", "down"}) { pbits[n] = meta.geti("bits"); pgroup[n] = group; }
  auto parse_pairs = [&](const std::string& key, std::map<std::string, int>& dst) {
    if (!meta.kv.count(key)) return;
    std::istringstream ss(meta.kv[key]);
    std::string tok;
    while (ss >> tok) { size_t c = tok.find(':'); dst[tok.substr(0, c)] = std::stoi(tok.substr(c + 1)); }
  };
  parse_pairs("proj_bits", pbits);
  parse_pairs("proj_group", pgroup);
  for (auto& [n, b] : pbits)
    if (b != 4 && b != 8) throw std::runtime_error(n + " projection is " + std::to_string(b) + "-bit; only 4 and 8 are supported");
  if (emu < 0) emu = meta.kv.count("dtype") && meta.kv["dtype"] == "bfloat16";
  std::vector<std::pair<int, int>> rounds;  // (id, rows)
  {
    std::istringstream ss(meta.kv["rounds"]);
    std::string tok;
    while (ss >> tok) {
      size_t c = tok.find(':');
      rounds.push_back({std::stoi(tok.substr(0, c)), std::stoi(tok.substr(c + 1))});
    }
  }
  std::printf("trace %s: %d experts, top-%d, hidden %d, intermediate %d, bits gate/up/down %d/%d/%d, %zu rounds, path %s, %s\n",
              dir.c_str(), E, topk, hidden, inter, pbits["gate"], pbits["up"], pbits["down"], rounds.size(),
              cpu ? "CPU" : "GPU", emu ? "bf16 rounding emulated" : "fp32 intermediates");
#ifdef __CUDACC__
  if (!cpu) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    std::printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
  }
#endif

  int failures = 0, checks = 0;
  std::printf("\n%-6s %-10s %6s %12s %12s %10s %10s  %s\n", "layer", "check", "rows", "max |err|", "max |ref|",
              "rel err", "cosine", "result");

  for (int L : meta.list("layers")) {
    const char* pn[3] = {"gate", "up", "down"};
    Npy q[3], s[3], b[3];
    QMat M[3];
    for (int p = 0; p < 3; p++) {
      std::string base = dir + "/weights/L" + std::to_string(L) + "_" + pn[p];
      int out = p < 2 ? inter : hidden, in = p < 2 ? hidden : inter;
      int bits = pbits[pn[p]], grp = pgroup[pn[p]];
      q[p] = load_npy(base + "_q.npy");
      s[p] = load_npy(base + "_s.npy");
      b[p] = load_npy(base + "_b.npy");
      check_shape(q[p], "<u4", {E, out, in * bits / 32}, base + "_q");
      check_shape(s[p], "<f4", {E, out, in / grp}, base + "_s");
      check_shape(b[p], "<f4", {E, out, in / grp}, base + "_b");
      M[p] = {q[p].as<uint32_t>(), s[p].as<float>(), b[p].as<float>(), E, out, in, grp, bits};
    }

#ifdef __CUDACC__
    QMat Md[3];
    if (!cpu)
      for (int p = 0; p < 3; p++)
        Md[p] = {to_dev<uint32_t>(q[p]), to_dev<float>(s[p]), to_dev<float>(b[p]), E, M[p].out, M[p].in, M[p].group, M[p].bits};
#endif

    // 1. dequantizer alone vs MLX's dequantization of expert ref_e
    for (int p = 0; p < 3; p++) {
      Npy ref = load_npy(dir + "/weights/L" + std::to_string(L) + "_" + pn[p] + "_e" + std::to_string(ref_e) + "_deq.npy");
      size_t n = (size_t)M[p].out * M[p].in;
      std::vector<float> got(n);
      if (cpu) {
        for (size_t t = 0; t < n; t++) got[t] = deq_one(M[p], ref_e, (int)(t / M[p].in), (int)(t % M[p].in));
      }
#ifdef __CUDACC__
      else {
        float* d;
        CK(cudaMalloc(&d, n * 4));
        k_dequant<<<(unsigned)((n + 255) / 256), 256>>>(Md[p], ref_e, d);
        CK(cudaGetLastError());
        CK(cudaMemcpy(got.data(), d, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaFree(d));
      }
#endif
      Err er = compare(got.data(), ref.as<float>(), n);
      bool ok = er.maxabs <= 1e-5 * (er.refmax + 1e-12);
      checks++;
      failures += !ok;
      std::printf("%-6d %-10s %6s %12.3e %12.3e %10.2e %10.6f  %s\n", L, (std::string("deq ") + pn[p]).c_str(),
                  "-", er.maxabs, er.refmax, er.maxabs / (er.refmax + 1e-30), er.cosine, ok ? "ok" : "FAIL");
    }

    // 2. every recorded round for this layer.
    //    strict: fp32 replay vs MLX's own float32 recompute (y32): pass/fail
    //    vs-mlx: replay (bf16 emulated if the model ran in bf16) vs what MLX
    //            actually produced on the Mac (y): reported, not pass/fail,
    //            because MLX backends differ in how they accumulate in bf16
    for (auto [rid, rows_meta] : rounds) {
      std::string base = dir + "/rounds/R" + std::to_string(rid) + "_L" + std::to_string(L);
      if (!file_exists(base + "_x.npy")) continue;
      Npy x = load_npy(base + "_x.npy"), id = load_npy(base + "_idx.npy"), yr = load_npy(base + "_y.npy");
      int rows = (int)x.shape[0];
      check_shape(x, "<f4", {rows, hidden}, base + "_x");
      check_shape(id, "<i4", {rows, topk}, base + "_idx");
      check_shape(yr, "<f4", {rows, topk, hidden}, base + "_y");
      for (size_t t = 0; t < id.count(); t++)
        if (id.as<int>()[t] < 0 || id.as<int>()[t] >= E) throw std::runtime_error(base + ": expert id out of range");
      bool has32 = file_exists(base + "_y32.npy");
      Npy y32;
      if (has32) {
        y32 = load_npy(base + "_y32.npy");
        check_shape(y32, "<f4", {rows, topk, hidden}, base + "_y32");
      }
      for (int mode = 0; mode < 2; mode++) {  // 0 = strict fp32, 1 = vs MLX output
        if (mode == 0 && !has32) continue;
        int e_mode = mode == 0 ? 0 : emu;
        std::vector<float> y((size_t)rows * topk * hidden);
        if (cpu) {
          moe_cpu(M[0], M[1], M[2], x.as<float>(), id.as<int>(), rows, topk, e_mode, y.data());
        }
#ifdef __CUDACC__
        else {
          float *dx, *dact, *dy;
          int* didx;
          CK(cudaMalloc(&dx, x.data.size()));
          CK(cudaMalloc(&didx, id.data.size()));
          CK(cudaMalloc(&dact, (size_t)rows * topk * inter * 4));
          CK(cudaMalloc(&dy, y.size() * 4));
          CK(cudaMemcpy(dx, x.data.data(), x.data.size(), cudaMemcpyHostToDevice));
          CK(cudaMemcpy(didx, id.data.data(), id.data.size(), cudaMemcpyHostToDevice));
          dim3 blk(32 * WARPS);
          k_gate_up<<<dim3((inter + WARPS - 1) / WARPS, rows * topk), blk>>>(Md[0], Md[1], dx, didx, topk, e_mode, dact);
          k_down<<<dim3((hidden + WARPS - 1) / WARPS, rows * topk), blk>>>(Md[2], dact, didx, e_mode, dy);
          CK(cudaGetLastError());
          CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
          CK(cudaFree(dx)); CK(cudaFree(didx)); CK(cudaFree(dact)); CK(cudaFree(dy));
        }
#endif
        const float* ref = mode == 0 ? y32.as<float>() : yr.as<float>();
        Err er = compare(y.data(), ref, y.size());
        double rel = er.maxabs / (er.refmax + 1e-30);
        std::string name = "r" + std::to_string(rid) + (mode == 0 ? " strict" : " vs-mlx");
        const char* verdict;
        if (mode == 0) {
          bool ok = rel <= strict_tol;
          checks++;
          failures += !ok;
          verdict = ok ? "ok" : "FAIL";
        } else {
          bool ok = rel <= tol && er.cosine >= cos_min;
          verdict = ok ? "info: close" : "info: differs";
        }
        std::printf("%-6d %-10s %6d %12.3e %12.3e %10.2e %10.6f  %s\n", L, name.c_str(), rows, er.maxabs,
                    er.refmax, rel, er.cosine, verdict);
      }
      if (!has32) std::printf("%-6d r%-9d no _y32.npy in this trace: re-export with the current spec_decode.py for the strict check\n", L, rid);
    }
#ifdef __CUDACC__
    if (!cpu)
      for (int p = 0; p < 3; p++) {
        CK(cudaFree((void*)Md[p].q)); CK(cudaFree((void*)Md[p].s)); CK(cudaFree((void*)Md[p].b));
      }
#endif
  }
  std::printf("\n%d/%d checks passed (dequant: rel err <= 1e-5; strict: fp32 replay vs MLX float32 recompute, rel err <= %.1e)\n",
              checks - failures, checks, strict_tol);
  std::printf("vs-mlx rows compare with MLX's actual bf16 output and are informational (close = rel err <= %.3g, cosine >= %.6g).\n",
              tol, cos_min);
  std::printf("rel err = max |replay - ref| / max |ref| over the whole round.\n");
  return failures ? 1 : 0;
}
