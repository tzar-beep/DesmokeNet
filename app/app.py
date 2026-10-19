"""RefDesmoke demo: runs locally in the browser on your own GPU.

    python app/app.py            then open http://127.0.0.1:7860

Image tab: smoky frame (+ optional clean reference) -> restored frame, confidence, gate map.
Video tab: CCTV clip -> side-by-side video [smoky | restored | confidence]. If no reference is
uploaded, the median of the first N frames is used, so the clip should start before the smoke.
"""
import sys
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from refdesmoke.infer import confidence_from_log_b, load_method, to_batch, to_u8  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_cache = {}


def available_methods():
    ckpts = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "runs").glob("*/best.pt"))
    return ["dcp"] + ckpts


def get_method(spec):
    if spec not in _cache:
        path = spec if spec == "dcp" else str(ROOT / spec)
        _cache[spec] = load_method(path, DEVICE)
    return _cache[spec]


def heatmap(map01, h, w):
    u8 = (np.clip(map01, 0, 1) * 255).astype(np.uint8)
    u8 = cv2.resize(u8, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def run_frame(method, smoky_u8, ref_u8):
    h, w = smoky_u8.shape[:2]
    smoky = to_batch(smoky_u8, DEVICE)
    ref = None
    if ref_u8 is not None:
        if ref_u8.shape[:2] != (h, w):
            ref_u8 = cv2.resize(ref_u8, (w, h), interpolation=cv2.INTER_AREA)
        ref = to_batch(ref_u8, DEVICE)
    res = method(smoky, ref)
    out = to_u8(res["out"])
    conf = gate = None
    if res["log_b"] is not None:
        conf = heatmap(confidence_from_log_b(res["log_b"])[0, 0].cpu().numpy(), h, w)
    if res["gate"] is not None:
        gate = heatmap(res["gate"][0, 0].float().cpu().numpy(), h, w)
    return out, conf, gate


def limit_width(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    return cv2.resize(img, (max_w, int(h * max_w / w) // 2 * 2), interpolation=cv2.INTER_AREA)


def image_tab(smoky, ref, spec, max_w):
    if smoky is None:
        raise gr.Error("Upload a smoky image.")
    method = get_method(spec)
    if method.needs_ref and ref is None:
        raise gr.Error("This model needs a clean reference frame of the same camera view.")
    smoky = limit_width(smoky, int(max_w))
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out, conf, gate = run_frame(method, smoky, ref)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0)
    info = f"{spec} on {DEVICE}: {ms:.1f} ms at {smoky.shape[1]}x{smoky.shape[0]}"
    return out, conf, gate, info


def open_writer(w, h, fps):
    tmp = Path(tempfile.mkdtemp())
    for name, fourcc in (("out.webm", "VP80"), ("out.mp4", "avc1"), ("out.mp4", "mp4v")):
        wr = cv2.VideoWriter(str(tmp / name), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if wr.isOpened():
            return wr, str(tmp / name)
    raise gr.Error("OpenCV could not open a video writer on this machine.")


def video_tab(video, ref, spec, n_ref, max_frames, max_w, progress=gr.Progress()):
    if video is None:
        raise gr.Error("Upload a video.")
    method = get_method(spec)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while len(frames) < int(max_frames):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(limit_width(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), int(max_w)))
    cap.release()
    if not frames:
        raise gr.Error("Could not read any frames from the video.")

    if ref is None and method.needs_ref:
        k = max(1, min(int(n_ref), len(frames)))
        ref = np.median(np.stack(frames[:k]), axis=0).astype(np.uint8)
        ref_note = f"reference = median of first {k} frames"
    else:
        ref_note = "reference = uploaded image" if ref is not None else "no reference used"

    h, w = frames[0].shape[:2]
    writer, path = open_writer(3 * w, h, fps)
    t0 = time.perf_counter()
    for f in progress.tqdm(frames, desc="desmoking"):
        out, conf, _ = run_frame(method, f, ref)
        conf = conf if conf is not None else np.zeros_like(f)
        writer.write(cv2.cvtColor(np.hstack([f, out, conf]), cv2.COLOR_RGB2BGR))
    writer.release()
    fps_proc = len(frames) / (time.perf_counter() - t0)
    info = f"{len(frames)} frames at {w}x{h}, {fps_proc:.1f} fps processing on {DEVICE}; {ref_note}"
    return path, path, (ref if ref is not None else None), info


def build_ui():
    with gr.Blocks(title="RefDesmoke") as ui:
        gr.Markdown("# RefDesmoke\nReference-guided, uncertainty-aware smoke removal for fixed cameras. "
                    "Confidence: bright = trust the pixel, dark = the model is unsure (likely hallucination).")
        with gr.Row():
            method = gr.Dropdown(available_methods(), value="dcp", label="Method")
            refresh = gr.Button("Refresh models", scale=0)
            max_w = gr.Slider(256, 1280, value=640, step=32, label="Max width (px)")
        refresh.click(lambda: gr.Dropdown(choices=available_methods()), outputs=method)

        with gr.Tab("Image"):
            with gr.Row():
                smoky = gr.Image(label="Smoky frame", type="numpy")
                ref = gr.Image(label="Clean reference (same camera, before the fire)", type="numpy")
            go = gr.Button("Desmoke", variant="primary")
            with gr.Row():
                out = gr.Image(label="Restored")
                conf = gr.Image(label="Confidence")
                gate = gr.Image(label="Reference validity gate (gated model only)")
            info = gr.Markdown()
            go.click(image_tab, [smoky, ref, method, max_w], [out, conf, gate, info])

        with gr.Tab("Video"):
            with gr.Row():
                video = gr.Video(label="CCTV clip (should start before the smoke)")
                vref = gr.Image(label="Optional clean reference", type="numpy")
            with gr.Row():
                n_ref = gr.Number(value=15, precision=0, label="Frames for auto reference")
                max_frames = gr.Number(value=300, precision=0, label="Max frames")
            vgo = gr.Button("Desmoke video", variant="primary")
            vout = gr.Video(label="Smoky | Restored | Confidence")
            vfile = gr.File(label="Download")
            used_ref = gr.Image(label="Reference used")
            vinfo = gr.Markdown()
            vgo.click(video_tab, [video, vref, method, n_ref, max_frames, max_w],
                      [vout, vfile, used_ref, vinfo])
    return ui


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1")
