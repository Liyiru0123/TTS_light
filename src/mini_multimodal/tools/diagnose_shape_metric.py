"""Diagnostic: is the P2-W04 shape *metric* faithful?

Renders clean reference shapes with the SAME renderer P2-W04 uses, then
runs the ACTUAL probe_image pipeline (component extraction + 64x64 resize +
P2-W01 shape classifier) to see whether the measurement agrees with ground
truth. Also isolates color leakage and the small-component discard threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.world.schema import (  # noqa: E402
    Color,
    Scene,
    SceneObject,
    Shape,
    Size,
)
from mini_multimodal.vision.renderer import render_scene  # noqa: E402
from mini_multimodal.ldm.artifacts import (  # noqa: E402
    SHAPE_CLASSES,
    _load_shape_model,
    probe_image,
)
from mini_multimodal.vision.schema import ImageRenderConfig  # noqa: E402


def build_scene(shape: Shape, color: Color, size: Size, scale: int = 2) -> Scene:
    obj = SceneObject(
        object_id="o1",
        shape=shape,
        color=color,
        size=size,
        initial_position=__import__("mini_multimodal.world.schema", fromlist=["Position"]).Position(0.5, 0.5),
        visible_start=0,
        visible_end=0,
    )
    return Scene(
        scene_id="diag",
        seed=0,
        canvas_width=64,
        canvas_height=64,
        frame_count=1,
        fps=1,
        objects=(obj,),
    )


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = torch.from_numpy(__import__("numpy").array(img, dtype="uint8")).permute(2, 0, 1).float() / 255.0
    return arr


def main() -> None:
    config = json = None
    # load config for shape checkpoint path
    import json as _json
    cfg = _json.loads((ROOT / "configs" / "p2_w04_tiny_image_ldm.json").read_text(encoding="utf-8"))
    shape_model = _load_shape_model(ROOT, cfg)

    shapes = list(Shape)
    colors = list(Color)
    sizes = [Size.SMALL, Size.MEDIUM, Size.LARGE]

    print("=" * 70)
    print("TEST 1: Clean reference shapes -> real probe_image pipeline")
    print("        (does the metric agree with ground truth on PERFECT renders?)")
    print("=" * 70)
    confusion = {}
    for shape in shapes:
        for color in colors:
            for size in sizes:
                scene = build_scene(shape, color, size)
                rendered = render_scene(scene, frame_index=0)
                img = rendered.image
                tensor = image_to_tensor(img)
                probe = probe_image(tensor, shape_model)
                pred_shape = probe["objects"][0]["shape"] if probe["objects"] else "NONE"
                key = (shape.value, color.value, size.value)
                confusion[key] = (pred_shape, probe["foreground_present"])
                ok = pred_shape == shape.value
                if not ok:
                    print(f"  MISMATCH gt={shape.value:9s} color={color.value:7s} size={size.value:7s} -> pred={pred_shape:9s} fg={probe['foreground_present']}")

    # aggregate per-shape accuracy on clean renders
    print("\n  Per-shape accuracy on CLEAN reference renders:")
    per_shape = {s.value: [0, 0] for s in shapes}
    for (gt, _, _), (pred, _) in confusion.items():
        per_shape[gt][1] += 1
        if pred == gt:
            per_shape[gt][0] += 1
    for s, (c, t) in per_shape.items():
        print(f"    {s:9s}: {c}/{t} = {c/t:.3f}")

    print("\n  Overall clean-reference shape accuracy: "
          f"{sum(v[0] for v in per_shape.values())}/{sum(v[1] for v in per_shape.values())}")

    print("\n" + "=" * 70)
    print("TEST 2: Color leakage — does COLOR change the predicted SHAPE?")
    print("        (render circle in every color; classifier should stay 'circle')")
    print("=" * 70)
    for shape in shapes:
        preds = []
        for color in colors:
            scene = build_scene(shape, color, Size.MEDIUM)
            rendered = render_scene(scene, frame_index=0)
            probe = probe_image(image_to_tensor(rendered.image), shape_model)
            preds.append(probe["objects"][0]["shape"] if probe["objects"] else "NONE")
        distinct = set(preds)
        flag = "  <-- COLOR LEAK!" if len(distinct) > 1 else ""
        print(f"  gt={shape.value:9s}: predicted across 6 colors = {sorted(distinct)}{flag}")

    print("\n" + "=" * 70)
    print("TEST 3: Small-shape discard threshold (len(pixels) >= 20 in _components)")
    print("=" * 70)
    for size in sizes:
        area_map = {Size.SMALL: "r=4px", Size.MEDIUM: "r=6px", Size.LARGE: "r=8px"}
        scene = build_scene(Shape.CIRCLE, Color.RED, size)
        rendered = render_scene(scene, frame_index=0)
        probe = probe_image(image_to_tensor(rendered.image), shape_model)
        n_comp = len(probe["objects"])
        print(f"  {size.value:7s} ({area_map[size]}): foreground_present={probe['foreground_present']}, "
              f"predicted_objects={n_comp}, pred={probe['objects'][0]['shape'] if n_comp else 'NONE'}")


if __name__ == "__main__":
    main()
