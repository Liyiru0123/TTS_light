# -*- coding: utf-8 -*-
"""仅从校准生成 LDM threshold_freeze.json（去原始检查点依赖，符合最终算法单步训练）。"""
import json, hashlib, sys
from pathlib import Path
from mini_multimodal.ldm.evaluation_artifacts import _derived_thresholds, _sha
def main():
    ROOT = Path(__file__).resolve().parents[3]
    cfg = json.loads((ROOT/"configs/p2_w04_tiny_image_ldm_v2.json").read_text(encoding="utf-8"))
    result_dir = ROOT/str(cfg["output_results"]); eval_dir = ROOT/str(cfg["output_eval_data"])
    cal = json.loads((result_dir/"evaluator_calibration.json").read_text(encoding="utf-8"))
    if not cal.get("verified"): raise RuntimeError("calibration must pass")
    proto = result_dir/"evaluator_shape_templates.npz"
    if not proto.exists(): raise RuntimeError("evaluator_shape_templates.npz 不存在（先跑 calibrate）")
    thresholds = _derived_thresholds(cal)
    freeze = {
        "schema_version": 1, "work_package": "P2-W04", "status": "frozen_before_model_revision",
        "date": "2026-09-06",
        "evaluation_manifest": str((eval_dir/"manifest.json").relative_to(ROOT)).replace("\\","/"),
        "evaluation_manifest_sha256": _sha(eval_dir/"manifest.json"),
        "calibration_file": str((result_dir/"evaluator_calibration.json").relative_to(ROOT)).replace("\\","/"),
        "calibration_file_sha256": _sha(result_dir/"evaluator_calibration.json"),
        "prototype_bank": str(proto.relative_to(ROOT)).replace("\\","/"),
        "prototype_bank_sha256": _sha(proto),
        "vae_upper_bounds": {
            "shape_accuracy": float(cal["vae"]["metrics"]["shape_accuracy"]["value"]),
            "normalized_template_iou_mean": float(cal["vae"]["metrics"]["normalized_template_iou_mean"]),
            "boundary_score_mean": float(cal["vae"]["metrics"]["boundary_score_mean"]),
        },
        "formula_policy": "Each derived metric is max(absolute_floor, documented_fraction * matching VAE upper bound). Other metrics use predeclared absolute thresholds.",
        "thresholds": thresholds,
        "statistics_policy": {"report_numerator_denominator_point_and_wilson_95": True,
                              "point_estimate_must_meet_threshold": True,
                              "wilson_lower_must_exceed_random_or_majority_baseline": True,
                              "per_shape_position_size_color_reporting_required": True},
        "checks": {"calibration_verified": cal["verified"], "thresholds_created_before_model_revision": True},
    }
    freeze["verified"] = all(freeze["checks"].values())
    fp = result_dir / "threshold_freeze.json"; fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"verified": freeze["verified"], "path": str(fp)}, ensure_ascii=False))
if __name__=="__main__": main()
