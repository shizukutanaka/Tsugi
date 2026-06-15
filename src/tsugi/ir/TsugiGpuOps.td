//===- TsugiGpuOps.td - Tsugi GPU dialect ------------------*- tablegen -*-===//
//
// Tsugi — Apache-2.0
//
// 中位 dialect。GPU実行モデル（block/warp/shared/barrier/matrix_mma）を表現。
// まだベンダー非依存。ここから vendor-split で NVVM / ROCDL へ分岐。
// SPEC.md §2.2 と1:1対応。
//
//===----------------------------------------------------------------------===//

#ifndef Tsugi_GPU_OPS
#define Tsugi_GPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tsugi_GpuDialect : Dialect {
  let name = "tsugi_gpu";
  let summary = "Tsugi mid-level GPU dialect (pre-vendor-split)";
  let cppNamespace = "::tsugi::gpu";
}

class Tsugi_GpuOp<string mnemonic, list<Trait> traits = []>
    : Op<Tsugi_GpuDialect, mnemonic, traits>;

// 抽象行列命令。vendor-split で各社intrinsicへlowering（ADR-004）。
def Tsugi_MatrixMmaOp : Tsugi_GpuOp<"matrix_mma", [Pure]> {
  let summary = "Abstract matrix multiply-accumulate at warp/wavefront scope";
  let description = [{
    shape 属性 = [M, N, K]（例 16x16x16）。lowering:
      NVIDIA → nvvm.wmma.mma.sync
      AMD    → rocdl.mfma.* (CDNA) / rocdl.wmma.* (RDNA3+)
  }];
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$c,
                       DenseI32ArrayAttr:$shape);
  let results = (outs AnyType:$result);
}

def Tsugi_BarrierOp : Tsugi_GpuOp<"barrier"> {
  let summary = "Block-level barrier (→ nvvm.barrier0 / rocdl.s.barrier)";
}

def Tsugi_AllocSharedOp : Tsugi_GpuOp<"alloc_shared"> {
  let summary = "Allocate shared/LDS memory (addrspace 3)";
  let results = (outs AnyMemRef:$result);
}

def Tsugi_LaneIdOp : Tsugi_GpuOp<"lane_id", [Pure]> {
  let summary = "Lane id within warp/wavefront";
  let results = (outs Index:$result);
}

#endif // Tsugi_GPU_OPS
