"""P3-W04 TinyT2V data preparation (DATA-VIDGEN-001/002).

- Extended vocabulary: P2-W04 image-condition vocab + motion tokens
- Trajectory model: speed profiles (constant/accelerate/decelerate/ease) + turns
  (two-segment direction change), delayed start, 2-object relative motion
- Videos: render_scene per frame with center_overrides (P1-W06 untouched)
- Text conditions: P1 describe_video format (video generation condition)
- Latents: P2-W02 image VAE encode per frame -> [8,4,16,16]
- Audit: splits, coverage, UNK rate, source integrity, counterfactual pairs

Evidence -> data/video/p3_w04_v1/
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.data import CharacterTokenizer  # noqa: E402
from mini_multimodal.tasks.reasoning import position_region  # noqa: E402
from mini_multimodal.tasks.text_zh import (  # noqa: E402
    COLOR_ZH, MOTION_ZH, REGION_ZH, SHAPE_ZH, SIZE_ZH,
)
from mini_multimodal.vae.model import TinyImageVAE  # noqa: E402
from mini_multimodal.video.artifacts import load_frames as load_p1_frames  # noqa: E402
from mini_multimodal.vision.renderer import render_scene  # noqa: E402
from mini_multimodal.vision.schema import ImageRenderConfig  # noqa: E402
from mini_multimodal.world.schema import (  # noqa: E402
    Color, MotionDirection, MotionSpec, Position, Scene, SceneObject, Shape, Size,
)

P1_VIDEO_DIR = ROOT / "data" / "video" / "p1_w06_v1"


def load_json(path: Path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------- trajectory model ----------------

PROFILE_FNS = {
    "constant": lambda t: t,
    "accelerate": lambda t: t * t,              # 先慢后快
    "decelerate": lambda t: 2 * t - t * t,      # 先快后慢
    "ease": lambda t: 3 * t * t - 2 * t * t * t,  # smoothstep 两端慢中间快
}

DIR_DELTA = {
    MotionDirection.LEFT: (-1.0, 0.0),
    MotionDirection.RIGHT: (1.0, 0.0),
    MotionDirection.UP: (0.0, -1.0),
    MotionDirection.DOWN: (0.0, 1.0),
}


def trajectory_positions(initial, speed, segments, frame_count: int) -> list[tuple[float, float]]:
    """Per-frame normalized positions from piecewise motion segments.

    segments: sorted non-overlapping (direction, start_frame, end_frame, profile).
    """
    positions = []
    for f in range(frame_count):
        x, y = initial
        for direction, start, end, profile in segments:
            if f < start:
                break
            steps = min(max(f - start, 0), end - start)
            if steps <= 0:
                continue
            duration = max(end - start, 1)
            frac = PROFILE_FNS[profile](steps / duration)
            seg_len = speed * duration
            dx, dy = DIR_DELTA[direction]
            x += dx * seg_len * frac
            y += dy * seg_len * frac
        positions.append((float(x), float(y)))
    return positions


def max_travel(speed: float, segments) -> float:
    return speed * max(1, sum(max(e - s, 1) for _, s, e, _ in segments))


def sample_object_spec(rng: np.random.Generator, config: dict, frame_count: int) -> dict:
    shapes = [s.value for s in Shape]
    colors = [c.value for c in Color]
    sizes = [z.value for z in Size]
    shape = str(rng.choice(shapes))
    color = str(rng.choice(colors))
    size = str(rng.choice(sizes))
    speed = float(rng.uniform(config["min_speed"], config["max_speed"]))
    profiles = config["trajectory_profiles"]
    is_static = bool(rng.random() < 0.25)
    segments = []
    if not is_static:
        directions = [MotionDirection.LEFT, MotionDirection.RIGHT, MotionDirection.UP, MotionDirection.DOWN]
        d1 = directions[int(rng.integers(4))]
        start1 = int(rng.integers(0, 4))
        end1 = int(rng.integers(start1 + 2, frame_count))
        profile1 = str(rng.choice(profiles))
        segments.append((d1, start1, end1, profile1))
        if rng.random() < config["turn_probability"] and end1 < frame_count - 1:
            d2 = directions[int(rng.integers(4))]
            end2 = frame_count - 1
            profile2 = str(rng.choice(profiles))
            segments.append((d2, end1, end2, profile2))
    return {"shape": shape, "color": color, "size": size, "speed": speed,
            "segments": segments, "is_static": is_static}


def initial_position_for(rng, speed, segments, frame_count, bounds=(0.05, 0.95)):
    travel = max_travel(speed, segments)
    lo, hi = bounds
    for _ in range(128):
        x = float(rng.uniform(lo + 0.02, hi - 0.02))
        y = float(rng.uniform(lo + 0.02, hi - 0.02))
        pos = trajectory_positions((x, y), speed, segments, frame_count)
        if all(lo <= px <= hi and lo <= py <= hi for px, py in pos):
            return (x, y), pos
    raise RuntimeError("unable to sample an in-bounds initial position")


def region_zh(x: float, y: float) -> str:
    return REGION_ZH[position_region(Position(x, y)).value]


def render_video_condition(object_specs: list[dict], frame_count: int) -> str:
    parts = []
    for i, spec in enumerate(object_specs, 1):
        base = (
            f"对象{i}是{COLOR_ZH[spec['color']]}的{SIZE_ZH[spec['size']]}"
            f"{SHAPE_ZH[spec['shape']]}，初始位于{region_zh(*spec['initial'])}"
        )
        visible = f"第{spec['visible_start'] + 1}帧至第{spec['visible_end'] + 1}帧可见"
        if spec["is_static"]:
            parts.append(f"{base}，保持静止，{visible}")
        else:
            seg_texts = [
                f"从第{s + 1}帧到第{e + 1}帧{MOTION_ZH[d.value]}"
                for d, s, e, _ in spec["segments"]
            ]
            parts.append(f"{base}，{'，'.join(seg_texts)}，{visible}")
    return "；".join(parts) + "。"


def pixel_center(pos, width: int, safe_margin: int) -> tuple[int, int]:
    return (
        round(safe_margin + pos[0] * (width - 1 - 2 * safe_margin)),
        round(safe_margin + pos[1] * (width - 1 - 2 * safe_margin)),
    )


def segments_to_json(segments) -> list[list]:
    return [[d.value, s, e, p] for d, s, e, p in segments]


def segments_from_json(segments) -> list:
    return [(MotionDirection(d), s, e, p) for d, s, e, p in segments]


# ---------------- vocabulary extension ----------------

NEW_PHRASES = ["初始位于", "从第", "帧到第", "帧至第", "保持静止", "可见",
               "向右移动", "向左移动", "向上移动", "向下移动"]
NEW_CHARS = ["初", "始", "从", "到", "向", "移", "动", "保", "持", "静", "止",
             "4", "5", "6", "7", "8"]


def build_extended_vocabulary(base_vocab: dict) -> dict:
    token_to_id = dict(base_vocab["token_to_id"])
    tokens = list(base_vocab["tokens"])
    for token in NEW_PHRASES + NEW_CHARS:
        if token not in token_to_id:
            token_to_id[token] = len(tokens)
            tokens.append(token)
    return {"token_to_id": token_to_id, "tokens": tokens}


# ---------------- rendering ----------------

def render_new_video(scene_id: str, spec: dict, render_cfg: ImageRenderConfig, video_cfg: dict):
    frame_count = int(spec["frame_count"])
    width = int(video_cfg["width"])
    safe_margin = int(video_cfg["safe_margin"])
    objects = []
    for i, s in enumerate(spec["specs"]):
        objects.append(SceneObject(
            object_id=f"obj_{i}",
            shape=Shape(s["shape"]), color=Color(s["color"]), size=Size(s["size"]),
            initial_position=Position(s["initial"][0], s["initial"][1]),
            motion=MotionSpec(direction=MotionDirection.STATIC, speed=0.0),
            visible_start=s["visible_start"], visible_end=s["visible_end"],
        ))
    scene = Scene(scene_id=scene_id, seed=0,
                  canvas_width=int(video_cfg["canvas_width"]), canvas_height=int(video_cfg["canvas_height"]),
                  frame_count=frame_count, fps=int(video_cfg["fps"]), objects=tuple(objects))
    frames = []
    for f in range(frame_count):
        overrides = {}
        for i, s in enumerate(spec["specs"]):
            if s["visible_start"] <= f <= s["visible_end"]:
                overrides[f"obj_{i}"] = pixel_center(s["trajectory"][f], width, safe_margin)
        rendered = render_scene(scene, frame_index=f, config=render_cfg, center_overrides=overrides)
        frames.append(np.asarray(rendered.image))
    return frames


def _encode_frames(vae, frames) -> torch.Tensor:
    batch = torch.stack([torch.from_numpy(fr).permute(2, 0, 1).float().div(255.0) for fr in frames])
    with torch.no_grad():
        mean, _ = vae.encode(batch)
    return mean


def _save_frames(frames_dir: Path, scene_id: str, frames) -> None:
    np.savez_compressed(frames_dir / f"{scene_id}.npz", frames=np.stack(frames).astype(np.uint8))


# ---------------- record emitters ----------------

def _emit_video(records, spec, split, index, out_dir, vae, render_cfg, video_cfg, frames_dir, latents_dir) -> int:
    scene_id = f"p3w04_{split}_{index:06d}"
    frames = render_new_video(scene_id, spec, render_cfg, video_cfg)
    prompt = render_video_condition(spec["specs"], spec["frame_count"])
    latent = _encode_frames(vae, frames)
    _save_frames(frames_dir, scene_id, frames)
    latents_path = latents_dir / f"{scene_id}.npz"
    np.savez_compressed(latents_path, latent=latent.numpy().astype(np.float16))
    records.append({
        "record_id": f"new:{split}:{scene_id}", "split": split, "scene_id": scene_id,
        "prompt": prompt, "source": "p3w04_new",
        "objects": [{
            "object_id": f"obj_{i}", "shape": s["shape"], "color": s["color"], "size": s["size"],
            "speed": s["speed"], "is_static": s["is_static"],
            "segments": segments_to_json(s["segments"]),
            "initial": list(s["initial"]), "trajectory": [list(p) for p in s["trajectory"]],
            "visible_start": s["visible_start"], "visible_end": s["visible_end"],
        } for i, s in enumerate(spec["specs"])],
        "latent_path": str(latents_path.relative_to(out_dir)).replace("\\", "/"),
        "frames_path": f"frames/{scene_id}.npz",
    })
    return 1


def _mirror_spec(spec: dict) -> dict:
    ms = copy.deepcopy(spec)
    for s in ms["specs"]:
        if s["is_static"]:
            continue
        new_segments = []
        for d, st, e, p in s["segments"]:
            nd = {MotionDirection.LEFT: MotionDirection.RIGHT,
                  MotionDirection.RIGHT: MotionDirection.LEFT}.get(d, d)
            new_segments.append((nd, st, e, p))
        s["segments"] = new_segments
        s["trajectory"] = trajectory_positions(s["initial"], s["speed"], new_segments, ms["frame_count"])
    return ms


def _in_bounds(spec: dict, lo: float = 0.02, hi: float = 0.98) -> bool:
    return all(
        all(lo <= px <= hi and lo <= py <= hi for px, py in s["trajectory"])
        for s in spec["specs"]
    )


def _min_separation_ok(spec: dict, width: int, safe_margin: int, min_px: float = 14.0) -> bool:
    if len(spec["specs"]) < 2:
        return True
    for f in range(spec["frame_count"]):
        centers = [pixel_center(s["trajectory"][f], width, safe_margin) for s in spec["specs"]]
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d2 = (centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2
                if d2 < min_px * min_px:
                    return False
    return True


def _emit_pair(records, spec, split, pair_index, out_dir, vae, render_cfg, video_cfg, frames_dir, latents_dir) -> int:
    mirrored = _mirror_spec(spec)
    n1 = _emit_video(records, spec, split, pair_index * 2, out_dir, vae, render_cfg, video_cfg, frames_dir, latents_dir)
    n2 = _emit_video(records, mirrored, split, pair_index * 2 + 1, out_dir, vae, render_cfg, video_cfg, frames_dir, latents_dir)
    records[-2]["counterfactual_group"] = f"cf_{pair_index:03d}"
    records[-2]["counterfactual_kind"] = "base"
    records[-1]["counterfactual_group"] = f"cf_{pair_index:03d}"
    records[-1]["counterfactual_kind"] = "mirrored"
    return n1 + n2


# ---------------- existing P1 videos ----------------

def _p1_records(annotations_path: Path, per_split: dict) -> tuple[list[dict], dict]:
    """Reuse P1-W06 videos (frames + annotation), derive prompts via video-condition format."""
    records = []
    counts = dict(per_split)
    frames_meta = {}
    with io.open(annotations_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            split = row["split"]
            if counts.get(split, 0) <= 0:
                continue
            counts[split] -= 1
            ann = row["annotation"]
            scene_id = ann["scene_id"]
            tracks = ann["tracks"]
            prompt = _p1_prompt(tracks)
            frames_meta[scene_id] = row["frames_path"]
            records.append({
                "record_id": f"existing:{split}:{scene_id}",
                "split": split, "scene_id": scene_id, "prompt": prompt,
                "source": "p1_w06",
                "objects": [{
                    "object_id": t["object_id"], "shape": t["shape"], "color": t["color"],
                    "size": t["size"], "speed": t["speed"],
                    "motion_direction": t["motion_direction"],
                    "visible_start": t["visible_start"], "visible_end": t["visible_end"],
                    "trajectory": [list(p["center_normalized"]) for p in t["points"]],
                } for t in tracks],
            })
    return records, frames_meta


def _p1_prompt(tracks: list[dict]) -> str:
    parts = []
    for i, t in enumerate(tracks, 1):
        base = (f"对象{i}是{COLOR_ZH[t['color']]}的{SIZE_ZH[t['size']]}"
                f"{SHAPE_ZH[t['shape']]}，初始位于{region_zh(*t['points'][0]['center_normalized'])}")
        visible = f"第{t['visible_start'] + 1}帧至第{t['visible_end'] + 1}帧可见"
        if t["motion_direction"] == "static":
            parts.append(f"{base}，保持静止，{visible}")
        else:
            start, end = _p1_motion_frames(t)
            parts.append(f"{base}，从第{start + 1}帧到第{end + 1}帧{MOTION_ZH[t['motion_direction']]}，{visible}")
    return "；".join(parts) + "。"


def _p1_motion_frames(track: dict) -> tuple[int, int]:
    centers = [p["center_normalized"] for p in track["points"]]
    n = len(centers)
    start = 0
    while start + 1 < n and np.allclose(centers[start], centers[start + 1], atol=1e-6):
        start += 1
    end = n - 1
    while end > start and np.allclose(centers[end], centers[end - 1], atol=1e-6):
        end -= 1
    return start, end


# ---------------- main ----------------

def main() -> int:
    cfg = load_json(ROOT / "configs" / "p3_w04_tiny_t2v.json")
    out_dir = ROOT / cfg["output_data"]
    out_dir.mkdir(parents=True, exist_ok=True)
    video_cfg = cfg["video"]
    gen_cfg = cfg["data_gen"]
    rng = np.random.default_rng(int(gen_cfg["seed"]))
    frame_count = int(video_cfg["frame_count"])
    render_cfg = ImageRenderConfig(output_scale=int(video_cfg["output_scale"]),
                                   safe_margin=int(video_cfg["safe_margin"]))
    width = int(video_cfg["width"])
    safe_margin = int(video_cfg["safe_margin"])
    frames_dir = out_dir / "frames"
    latents_dir = out_dir / "latents"
    frames_dir.mkdir(parents=True, exist_ok=True)
    latents_dir.mkdir(parents=True, exist_ok=True)

    # ---- extended vocabulary ----
    package = torch.load(ROOT / cfg["image_ldm_package"], map_location="cpu", weights_only=False)
    vocab = build_extended_vocabulary(package["vocabulary"])
    dump_json(out_dir / "vocabulary.json", vocab)
    tokenizer = CharacterTokenizer(vocab)
    print(f"vocabulary: {len(vocab['tokens'])} tokens (base {len(package['vocabulary']['tokens'])} + {len(NEW_PHRASES + NEW_CHARS)})")

    # ---- VAE ----
    vae_cfg = load_json(ROOT / cfg["image_vae_config"])
    vae = TinyImageVAE(**{key: vae_cfg["model"][key] for key in (
        "base_channels", "latent_channels", "logvar_min", "logvar_max")})
    ckpt = torch.load(ROOT / cfg["image_vae_checkpoint"], map_location="cpu", weights_only=False)
    vae.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt, strict=True)
    vae.eval()

    # ---- existing P1 videos ----
    existing_per_split = {"train": 1200, "validation": 200, "test": 200}
    p1_records, p1_frames_meta = _p1_records(ROOT / cfg["source_p1_video_annotations"], existing_per_split)
    records = list(p1_records)
    print(f"existing P1 records: {len(p1_records)}")

    # encode latents for P1 frames (reuse stored frames)
    for r in records:
        frames = load_p1_frames(P1_VIDEO_DIR / p1_frames_meta[r["scene_id"]])  # [8,64,64,3]
        latent = _encode_frames(vae, [frames[f] for f in range(frame_count)])
        latents_path = latents_dir / f"{r['scene_id']}.npz"
        np.savez_compressed(latents_path, latent=latent.numpy().astype(np.float16))
        r["latent_path"] = str(latents_path.relative_to(out_dir)).replace("\\", "/")
        r["frames_path"] = p1_frames_meta[r["scene_id"]]

    # ---- new videos ----
    new_per_split = {k: int(gen_cfg["split_counts"][k]) - v for k, v in existing_per_split.items()}
    counterfactual_target = int(gen_cfg["counterfactual_pair_count"])
    cf_created = 0
    for split, count in new_per_split.items():
        made = 0
        attempts = 0
        while made < count:
            attempts += 1
            if attempts > count * 50:
                raise RuntimeError(f"split {split} stuck after {attempts} attempts")
            spec = {"specs": [], "frame_count": frame_count}
            n_objects = int(rng.integers(1, 3))
            for _ in range(n_objects):
                ospec = sample_object_spec(rng, gen_cfg, frame_count)
                ospec["initial"], ospec["trajectory"] = initial_position_for(
                    rng, ospec["speed"], ospec["segments"], frame_count)
                ospec["visible_start"] = 0
                ospec["visible_end"] = frame_count - 1
                spec["specs"].append(ospec)
            if not _in_bounds(spec) or not _min_separation_ok(spec, width, safe_margin):
                continue
            if split == "test" and cf_created < counterfactual_target:
                mirrored = _mirror_spec(spec)
                if not _in_bounds(mirrored):
                    continue
                made += _emit_pair(records, spec, split, cf_created, out_dir, vae, render_cfg,
                                   video_cfg, frames_dir, latents_dir)
                cf_created += 1
            else:
                made += _emit_video(records, spec, split, len(records), out_dir, vae, render_cfg,
                                    video_cfg, frames_dir, latents_dir)
    print(f"total records: {len(records)} (counterfactual pairs: {cf_created})")

    # ---- write records + manifest ----
    with io.open(out_dir / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    audit = _audit(records, tokenizer, cfg)
    dump_json(out_dir / "data_audit.json", audit)
    dump_json(out_dir / "temporal_pair_audit.json", audit["temporal_pairs"])
    manifest = {
        "schema_version": 1, "work_package": "P3-W04", "artifact_version": 1,
        "video_count": len(records), "frame_count": frame_count,
        "width": int(video_cfg["width"]), "height": int(video_cfg["height"]),
        "fps": int(video_cfg["fps"]),
        "split_counts": {s: sum(1 for r in records if r["split"] == s)
                         for s in ("train", "validation", "test")},
        "trajectory_coverage": audit["trajectory_coverage"],
        "source_p1_manifest_sha256": audit["source_p1_manifest_sha256"],
    }
    dump_json(out_dir / "manifest.json", manifest)
    print("split counts:", json.dumps(manifest["split_counts"], ensure_ascii=False))
    print("audit:", json.dumps({k: v for k, v in audit.items() if k != "temporal_pairs"},
                               ensure_ascii=False)[:800])
    print(f"evidence -> {out_dir}")
    return 0


def _audit(records: list[dict], tokenizer: CharacterTokenizer, cfg: dict) -> dict:
    split_sets = {s: set() for s in ("train", "validation", "test")}
    unk_count = 0
    coverage = {"constant": 0, "accelerate": 0, "decelerate": 0, "ease": 0,
                "turn": 0, "delayed_start": 0, "two_object": 0, "static": 0}
    unk_id = tokenizer.token_to_id["<unk>"]
    for r in records:
        split_sets[r["split"]].add(r["scene_id"])
        ids = tokenizer.encode(r["prompt"], 128)
        if unk_id in ids:
            unk_count += 1
        if r["source"] == "p3w04_new":
            for o in r["objects"]:
                if o["is_static"]:
                    coverage["static"] += 1
                for _d, s, _e, p in o["segments"]:
                    coverage[p] += 1
                    if s > 0:
                        coverage["delayed_start"] += 1
                if len(o["segments"]) > 1:
                    coverage["turn"] += 1
            if len(r["objects"]) > 1:
                coverage["two_object"] += 1
    pairs = {}
    for r in records:
        if r.get("counterfactual_group"):
            pairs.setdefault(r["counterfactual_group"], []).append(r)
    pair_ok = all(len(v) == 2 and {x["counterfactual_kind"] for x in v} == {"base", "mirrored"}
                  for v in pairs.values())
    # P1 source integrity: annotations.jsonl hash vs recorded in P1 manifest
    p1_manifest = load_json(ROOT / cfg["source_p1_video_manifest"])
    ann_sha = hashlib.sha256((ROOT / cfg["source_p1_video_annotations"]).read_bytes()).hexdigest()
    recorded = p1_manifest["files"].get("annotations.jsonl", {})
    ann_integrity = recorded.get("sha256") is None or recorded["sha256"] == ann_sha
    return {
        "train_video_count": len(split_sets["train"]),
        "validation_video_count": len(split_sets["validation"]),
        "test_video_count": len(split_sets["test"]),
        "split_intersections": {
            "train_val": len(split_sets["train"] & split_sets["validation"]),
            "train_test": len(split_sets["train"] & split_sets["test"]),
            "val_test": len(split_sets["validation"] & split_sets["test"]),
        },
        "unk_prompt_count": unk_count,
        "trajectory_coverage": coverage,
        "temporal_pairs": {"count": len(pairs), "structure_ok": pair_ok,
                           "pair_ids": sorted(pairs.keys())},
        "source_p1_manifest_sha256": hashlib.sha256(
            io.open(cfg["source_p1_video_manifest"], "rb").read()).hexdigest(),
        "source_p1_annotations_integrity": ann_integrity,
    }


if __name__ == "__main__":
    raise SystemExit(main())
