"""Step-1 UI for TRELLIS2 sparse-structure drag editing.

This file only defines Gradio inputs for:
1) source points
2) target points
3) user-customized edit mask

No pipeline/model logic is included yet.
"""

from __future__ import annotations

import json
from typing import List, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw


Point = Tuple[int, int]


def _draw_points(image: np.ndarray | None, src_points: List[Point], tgt_points: List[Point]) -> Image.Image:
    """Draw source/target points on top of the current image."""
    if image is None:
        image = np.zeros((512, 512, 3), dtype=np.uint8)

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    canvas = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    radius = 5
    for x, y in src_points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 64, 64), outline=(255, 255, 255))

    for x, y in tgt_points:
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=(64, 180, 255), outline=(255, 255, 255))

    for (sx, sy), (tx, ty) in zip(src_points, tgt_points):
        draw.line((sx, sy, tx, ty), fill=(255, 255, 0), width=2)

    return canvas


def _on_click(
    evt: gr.SelectData,
    base_image: np.ndarray | None,
    src_points: List[Point],
    tgt_points: List[Point],
) -> tuple[Image.Image, List[Point], List[Point], str, str]:
    """Alternate click assignment: source point then target point."""
    x, y = int(evt.index[0]), int(evt.index[1])

    if len(src_points) == len(tgt_points):
        src_points = src_points + [(x, y)]
    else:
        tgt_points = tgt_points + [(x, y)]

    preview = _draw_points(base_image, src_points, tgt_points)
    return (
        preview,
        src_points,
        tgt_points,
        json.dumps(src_points, ensure_ascii=False),
        json.dumps(tgt_points, ensure_ascii=False),
    )


def _clear_points(base_image: np.ndarray | None) -> tuple[Image.Image, list, list, str, str]:
    preview = _draw_points(base_image, [], [])
    return preview, [], [], "[]", "[]"


def _extract_mask(image_editor_value) -> np.ndarray | None:
    """Extract a binary mask from gr.ImageEditor value."""
    if image_editor_value is None:
        return None

    # gr.ImageEditor value is commonly a dict with background/layers/composite
    mask_img = None
    if isinstance(image_editor_value, dict):
        if image_editor_value.get("composite") is not None:
            mask_img = image_editor_value["composite"]
        elif image_editor_value.get("layers"):
            mask_img = image_editor_value["layers"][-1]
    else:
        mask_img = image_editor_value

    if mask_img is None:
        return None

    if isinstance(mask_img, Image.Image):
        arr = np.array(mask_img)
    else:
        arr = np.array(mask_img)

    if arr.ndim == 3 and arr.shape[-1] == 4:
        alpha = arr[..., 3]
        return (alpha > 0).astype(np.uint8)

    if arr.ndim == 3:
        gray = arr.mean(axis=-1)
        return (gray > 0).astype(np.uint8)

    return (arr > 0).astype(np.uint8)


def _preview_mask(image_editor_value) -> Image.Image | None:
    mask = _extract_mask(image_editor_value)
    if mask is None:
        return None

    vis = (mask * 255).astype(np.uint8)
    return Image.fromarray(vis, mode="L")


with gr.Blocks(title="TRELLIS2 Sparse Structure Drag - Step 1") as demo:
    gr.Markdown(
        """
# TRELLIS2 Sparse Structure Drag Editing (Step 1)

- 在左侧图上点击设置点位（交替记录为 **source -> target**）。
- 在 Mask Editor 中涂抹定义编辑区域。
- 当前仅实现输入界面，后续步骤会接入模型/管线/sampler。
        """
    )

    src_state = gr.State([])
    tgt_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            click_image = gr.Image(label="Click Image (source/target points)", type="numpy", height=420)
            clear_btn = gr.Button("Clear Points")
            src_json = gr.Textbox(label="Source Points (JSON)", value="[]")
            tgt_json = gr.Textbox(label="Target Points (JSON)", value="[]")

        with gr.Column(scale=1):
            mask_editor = gr.ImageEditor(
                label="Input Mask Editor (paint editable region)",
                type="pil",
                brush=gr.Brush(colors=["#ffffff"], default_color="#ffffff", color_mode="fixed"),
                height=420,
            )
            mask_preview = gr.Image(label="Binary Mask Preview", type="pil", image_mode="L")

    click_image.select(
        _on_click,
        inputs=[click_image, src_state, tgt_state],
        outputs=[click_image, src_state, tgt_state, src_json, tgt_json],
    )
    clear_btn.click(_clear_points, inputs=[click_image], outputs=[click_image, src_state, tgt_state, src_json, tgt_json])

    mask_editor.change(_preview_mask, inputs=[mask_editor], outputs=[mask_preview])


if __name__ == "__main__":
    demo.launch()
