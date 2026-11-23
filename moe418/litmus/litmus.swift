// Metal memory-model litmus tests for the cross-threadgroup handoffs the fused MoE kernel relies on.
//
//   swiftc -O litmus.swift -o litmus && ./litmus [iterations]      (default 200)
//
// Each test runs many independent instances at once. Role A threads sit in even threadgroups, role B in odd
// ones, and B's lane is permuted so a pair never shares a threadgroup or a SIMD group. Each instance gets its
// own memory locations, 128 bytes apart, so x and y never share a cache line. We count how often each outcome
// appears; a "weak" outcome is one sequential consistency forbids.
//
//   MP  message passing    A: x=1; [F]; y=1          B: r0=y; [F]; r1=x     weak: r0=1, r1=0
//   SB  store buffering    A: x=1; [F]; r0=y         B: y=1; [F]; r1=x      weak: r0=0, r1=0
//   LB  load buffering     A: r0=y; [F]; x=1         B: r1=x; [F]; y=1      weak: r0=1, r1=1
//   MPd message passing with x as plain (non-atomic) data: plain, then coherent(device)
//
// [F] is either nothing or atomic_thread_fence(mem_device, seq_cst, thread_scope_device).
// All atomic accesses are relaxed, the only order Metal offers.
import Foundation
import Metal

let header = """
#include <metal_stdlib>
using namespace metal;
#define RLX memory_order_relaxed
#define FENCE atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)
#define NOFENCE

#define GRID uint wg [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], \\
             uint tpg [[threads_per_threadgroup]]
#define SETUP uint role = wg & 1u; \\
              uint inst = (wg >> 1) * tpg + (role == 0 ? lid : (lid * 7u + 3u) % tpg); \\
              device atomic_uint* x = mem + inst * stride; \\
              device atomic_uint* y = x + stride / 2;

#define ARGS device atomic_uint* mem [[buffer(0)]], device uint* res [[buffer(1)]], \\
             constant uint& stride [[buffer(3)]], GRID

#define MP(name, F) kernel void name(ARGS) { SETUP \\
  if (role == 0) { atomic_store_explicit(x, 1u, RLX); F; atomic_store_explicit(y, 1u, RLX); } \\
  else { uint r0 = atomic_load_explicit(y, RLX); F; uint r1 = atomic_load_explicit(x, RLX); \\
         res[2 * inst] = r0; res[2 * inst + 1] = r1; } }

#define SB(name, F) kernel void name(ARGS) { SETUP \\
  if (role == 0) { atomic_store_explicit(x, 1u, RLX); F; res[2 * inst] = atomic_load_explicit(y, RLX); } \\
  else { atomic_store_explicit(y, 1u, RLX); F; res[2 * inst + 1] = atomic_load_explicit(x, RLX); } }

#define LB(name, F) kernel void name(ARGS) { SETUP \\
  if (role == 0) { uint r0 = atomic_load_explicit(y, RLX); F; atomic_store_explicit(x, 1u, RLX); res[2 * inst] = r0; } \\
  else { uint r1 = atomic_load_explicit(x, RLX); F; atomic_store_explicit(y, 1u, RLX); res[2 * inst + 1] = r1; } }

MP(mp_nofence, NOFENCE)
MP(mp_fence, FENCE)
SB(sb_nofence, NOFENCE)
SB(sb_fence, FENCE)
LB(lb_nofence, NOFENCE)
LB(lb_fence, FENCE)
"""

// MP with the payload as plain data; QUAL is either empty or coherent(device).
func mpData(_ qual: String) -> String {
    let n = qual.isEmpty ? "plain" : "coherent"
    var s = ""
    for (fname, f) in [("nofence", "NOFENCE"), ("fence", "FENCE")] {
        s += """

        kernel void mpd_\(n)_\(fname)(ARGS, \(qual) device uint* data [[buffer(2)]]) { SETUP
          \(qual) device uint* d = data + inst * stride;
          if (role == 0) { *d = 1u; \(f); atomic_store_explicit(y, 1u, RLX); }
          else { uint r0 = atomic_load_explicit(y, RLX); \(f); uint r1 = *d;
                 res[2 * inst] = r0; res[2 * inst + 1] = r1; } }
        """
    }
    return s
}

guard let dev = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
let queue = dev.makeCommandQueue()!
let iters = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1])! : 200
let pairGroups = 64, tpgWanted = 256, stride: UInt32 = 64        // 64 uints = 256 B per instance
print("device: \(dev.name), iterations: \(iters)\n")

func compile(_ src: String, _ what: String) -> MTLLibrary? {
    do { return try dev.makeLibrary(source: src, options: nil) }
    catch { print("[\(what)] does not compile: \(error.localizedDescription.prefix(300))\n"); return nil }
}

typealias Weak = (UInt32, UInt32) -> Bool
let mpWeak: Weak = { $0 == 1 && $1 == 0 }
var tests: [(String, Weak, MTLLibrary)] = []
if let lib = compile(header, "atomics") {
    for t in ["mp_nofence", "mp_fence"] { tests.append((t, mpWeak, lib)) }
    for t in ["sb_nofence", "sb_fence"] { tests.append((t, { $0 == 0 && $1 == 0 }, lib)) }
    for t in ["lb_nofence", "lb_fence"] { tests.append((t, { $0 == 1 && $1 == 1 }, lib)) }
}
if let lib = compile(header + mpData(""), "plain data") {
    for t in ["mpd_plain_nofence", "mpd_plain_fence"] { tests.append((t, mpWeak, lib)) }
}
// Spelling of the coherence qualifier has varied; try both orders.
if let lib = compile(header + mpData("coherent(device)"), "coherent(device), qualifier first")
    ?? compile(header + mpData("coherent(device)").replacingOccurrences(
        of: "coherent(device) device", with: "device coherent(device)"), "coherent(device), qualifier second") {
    for t in ["mpd_coherent_nofence", "mpd_coherent_fence"] { tests.append((t, mpWeak, lib)) }
}

print("| test | instances run | weak outcomes | weak rate | outcomes (r0r1: count) |")
print("|---|---|---|---|---|")
for (name, weak, lib) in tests {
    let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    var tpg = tpgWanted
    while tpg > pso.maxTotalThreadsPerThreadgroup { tpg /= 2 }
    let n = pairGroups * tpg
    let bytes = n * Int(stride) * 4
    let mem = dev.makeBuffer(length: bytes, options: .storageModeShared)!
    let data = dev.makeBuffer(length: bytes, options: .storageModeShared)!
    let res = dev.makeBuffer(length: n * 2 * 4, options: .storageModeShared)!
    var stride_ = stride
    var hist: [String: Int] = [:]
    var weakCount = 0
    for _ in 0..<iters {
        memset(mem.contents(), 0, bytes); memset(data.contents(), 0, bytes); memset(res.contents(), 0xFF, n * 8)
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(mem, offset: 0, index: 0)
        enc.setBuffer(res, offset: 0, index: 1)
        enc.setBuffer(data, offset: 0, index: 2)
        enc.setBytes(&stride_, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: 2 * pairGroups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { fatalError("\(name): \(e)") }
        let r = res.contents().bindMemory(to: UInt32.self, capacity: n * 2)
        for i in 0..<n {
            let (a, b) = (r[2 * i], r[2 * i + 1])
            hist["\(a)\(b)", default: 0] += 1
            if weak(a, b) { weakCount += 1 }
        }
    }
    let total = iters * n
    let h = hist.sorted { $0.key < $1.key }.map { "\($0.key): \($0.value)" }.joined(separator: ", ")
    print("| \(name) | \(total) | \(weakCount) | \(String(format: "%.2e", Double(weakCount) / Double(total))) | \(h) |")
}
