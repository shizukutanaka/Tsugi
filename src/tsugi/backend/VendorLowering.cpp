//===- VendorLowering.cpp - tsugi.gpu → NVVM / ROCDL ----------------------===//
//
// Tsugi — Apache-2.0
//
// vendor-split パス。tsugi.gpu dialect を各社へ lowering する分岐点。
// ARCHITECTURE.md §3 / SPEC.md §2.3 / §3 に対応。
//
// 状態: 骨格（skeleton）。実 lowering パターンは Phase 1-2 で実装。
//       実 GPU での動作検証は NVIDIA・AMD 実機が必要（未検証＝未検証と明記）。
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

namespace tsugi {

enum class Vendor { NVIDIA, AMD, SPIRV /* fallback: Intel/Vulkan */ };

/// tsugi.gpu.matrix_mma を各社の行列コア intrinsic へ lowering する。
/// ADR-004: Vulkan cooperative matrix には依存しない。MLIR intrinsic 直叩き。
///
///   NVIDIA  : nvvm.wmma.mma.sync          (NVPTX が LLVM intrinsic として公開)
///   AMD CDNA: rocdl.mfma.f32.16x16x16f16
///   AMD RDNA: rocdl.wmma.*
///
/// TODO(Phase2): 各社の thread-data layout を autotune スケジュールと接続。
///   世代間で挙動変化（HMMA.884→16816）するため target arch を明示要求。
class VendorLoweringPass {
public:
  explicit VendorLoweringPass(Vendor v) : vendor(v) {}

  /// 中位 tsugi.gpu モジュールを受け取り、ターゲット dialect へ書き換える。
  /// 返り値: 成功可否。失敗時は理由を診断として出す（捏造しない）。
  bool run(mlir::ModuleOp module) {
    switch (vendor) {
    case Vendor::NVIDIA:
      // TODO(Phase1): tsugi.gpu → NVVM dialect 変換パターンを登録。
      //   matrix_mma → nvvm.wmma, barrier → nvvm.barrier0, lane_id → laneid
      llvm::errs() << "[tsugi] NVIDIA lowering: not yet implemented (Phase1)\n";
      return false;
    case Vendor::AMD:
      // TODO(Phase1): tsugi.gpu → ROCDL dialect 変換パターンを登録。
      //   matrix_mma → rocdl.mfma/wmma, barrier → rocdl.s.barrier
      llvm::errs() << "[tsugi] AMD lowering: not yet implemented (Phase1)\n";
      return false;
    case Vendor::SPIRV:
      // フォールバックのみ（Intel/Vulkan・v1.0以降）。ADR-001。
      llvm::errs() << "[tsugi] SPIR-V fallback: out of v0.1 scope\n";
      return false;
    }
    return false;
  }

private:
  Vendor vendor;
};

} // namespace tsugi
