//===- TsugiTileOps.td - Tsugi Tile dialect ----------------*- tablegen -*-===//
//
// Tsugi — Apache-2.0
//
// 高位タイル演算 dialect。ベンダー非依存。SPEC.md §2.1 と1:1対応。
// ユーザーDSL / torch.compile が生成する最初のIR。
//
//===----------------------------------------------------------------------===//

#ifndef Tsugi_TILE_OPS
#define Tsugi_TILE_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tsugi_TileDialect : Dialect {
  let name = "tsugi_tile";
  let summary = "Tsugi high-level tile dialect (vendor-agnostic)";
  let cppNamespace = "::tsugi::tile";
}

class Tsugi_TileOp<string mnemonic, list<Trait> traits = []>
    : Op<Tsugi_TileDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// tile.dot : 行列積（行列コアへlowering・R1の核・ADR-004）
//===----------------------------------------------------------------------===//
def Tsugi_DotOp : Tsugi_TileOp<"dot", [Pure]> {
  let summary = "Tiled matrix multiply-accumulate (lowers to vendor matrix cores)";
  let description = [{
    `acc = dot(a, b) + acc`. lowering 先:
      NVIDIA → nvvm.wmma.mma.sync
      AMD CDNA → rocdl.mfma.* / RDNA3+ → rocdl.wmma.*
    Vulkan cooperative matrix には依存しない（ADR-004）。
  }];
  let arguments = (ins AnyRankedTensor:$a,
                       AnyRankedTensor:$b,
                       AnyRankedTensor:$acc);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
    "$a `,` $b `,` $acc attr-dict `:` type($a) `,` type($b) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// tile.load / tile.store : グローバル↔タイル
//===----------------------------------------------------------------------===//
def Tsugi_LoadOp : Tsugi_TileOp<"load"> {
  let summary = "Load a tile from a global pointer with coalescing hints";
  let arguments = (ins AnyType:$ptr, Variadic<Index>:$offsets);
  let results = (outs AnyRankedTensor:$result);
}

def Tsugi_StoreOp : Tsugi_TileOp<"store"> {
  let summary = "Store a tile to a global pointer";
  let arguments = (ins AnyType:$ptr, Variadic<Index>:$offsets, AnyRankedTensor:$value);
}

//===----------------------------------------------------------------------===//
// tile.reduce : 軸reduce（softmax/norm用）
//===----------------------------------------------------------------------===//
def Tsugi_ReduceOp : Tsugi_TileOp<"reduce", [Pure]> {
  let summary = "Reduce along an axis (sum/max) — lowers to warp/wavefront reduce";
  let arguments = (ins AnyRankedTensor:$input, I32Attr:$axis, StrAttr:$kind);
  let results = (outs AnyRankedTensor:$result);
}

#endif // Tsugi_TILE_OPS
