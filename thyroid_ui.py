#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import gradio as gr
from typing import Optional, Dict, Literal, Tuple


# =========================
# Helpers
# =========================

def _to_uint8(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None
    if np.issubdtype(img.dtype, np.floating):
        if float(np.nanmax(img)) <= 1.5:
            img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def to_gray_uint8(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None
    img = _to_uint8(img)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.astype(np.uint8)


def ensure_rgb_uint8(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None
    img = _to_uint8(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def adjust_saturation(rgb_u8: np.ndarray, factor: float) -> np.ndarray:
    factor = float(np.clip(factor, 0.0, 5.0))
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def ref_has_chroma(ref_bgr_u8: np.ndarray, sat_thresh: float = 6.0) -> bool:
    hsv = cv2.cvtColor(ref_bgr_u8, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    mask = v > 20
    if mask.sum() < 100:
        return False
    return float(s[mask].mean()) >= sat_thresh


# =========================
# Preprocess (thyroid US)
# =========================

def _largest_cc_mask(bin_mask_u8: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask_u8, connectivity=8)
    if num <= 1:
        return bin_mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(bin_mask_u8)
    out[labels == idx] = 255
    return out


def make_ultrasound_scan_mask(gray_u8: np.ndarray, thresh: int = 8) -> np.ndarray:
    m = (gray_u8 > thresh).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k2, iterations=1)
    m = _largest_cc_mask(m)

    h, w = m.shape
    flood = m.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    m = cv2.bitwise_or(m, holes)
    return m


def despeckle_ultrasound(gray_u8: np.ndarray, strength: int = 7) -> np.ndarray:
    strength = int(np.clip(strength, 0, 20))
    if strength <= 0:
        return gray_u8
    k = 3 if strength < 6 else 5
    x = cv2.medianBlur(gray_u8, k)
    d = 5
    sigma_color = 20 + 6 * strength
    sigma_space = 20 + 4 * strength
    x = cv2.bilateralFilter(x, d=d, sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))
    return x


def clahe_ultrasound(gray_u8: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    clip_limit = float(np.clip(clip_limit, 0.1, 10.0))
    tile = int(np.clip(tile, 4, 32))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray_u8)


def preprocess_thyroid_ultrasound(
    gray_u8: np.ndarray,
    mask_thresh: int = 8,
    despeckle_strength: int = 7,
    do_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = make_ultrasound_scan_mask(gray_u8, thresh=mask_thresh)
    x = despeckle_ultrasound(gray_u8, strength=despeckle_strength)

    if do_clahe:
        x2 = x.copy()
        x2[mask == 0] = 0
        x2 = clahe_ultrasound(x2, clip_limit=clahe_clip, tile=clahe_tile)
        x2[mask == 0] = 0
        x = x2
    else:
        x[mask == 0] = 0

    return x, mask


def apply_mask_rgb(rgb_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    out = rgb_u8.copy()
    out[mask_u8 == 0] = 0
    return out


# =========================
# Colorization methods (NO region-based)
# =========================

def create_gradient_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.float32)
    for i in range(256):
        t = i / 255.0
        lut[i] = [t, 1 - abs(2 * t - 1), 1 - t]
    return (lut * 255).astype(np.uint8)


LUT_GRADIENT = create_gradient_lut()

OPENCV_CMAPS: Dict[str, int] = {
    "TURBO": cv2.COLORMAP_TURBO,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "INFERNO": cv2.COLORMAP_INFERNO,
}


def apply_lut_rgb(gray_u8: np.ndarray, lut_rgb_u8: np.ndarray) -> np.ndarray:
    return lut_rgb_u8[gray_u8]


def apply_opencv_colormap_rgb(gray_u8: np.ndarray, cmap_name: str) -> np.ndarray:
    cmap = OPENCV_CMAPS.get(cmap_name.upper(), cv2.COLORMAP_TURBO)
    bgr = cv2.applyColorMap(gray_u8, cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def thyroid_quantile_band_colorization(gray_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    vals = gray_u8[mask_u8 > 0]
    if vals.size < 100:
        out = np.zeros((*gray_u8.shape, 3), dtype=np.uint8)
        out[gray_u8 < 85] = [0, 0, 150]
        out[(gray_u8 >= 85) & (gray_u8 < 170)] = [255, 255, 0]
        out[gray_u8 >= 170] = [255, 0, 0]
        return out

    t1 = int(np.percentile(vals, 33))
    t2 = int(np.percentile(vals, 66))
    out = np.zeros((*gray_u8.shape, 3), dtype=np.uint8)
    out[gray_u8 < t1] = [30, 60, 200]
    out[(gray_u8 >= t1) & (gray_u8 < t2)] = [220, 220, 40]
    out[gray_u8 >= t2] = [220, 60, 30]
    return out


def kmeans_texture_colorization(gray_u8: np.ndarray, mask_u8: np.ndarray, k: int = 4) -> np.ndarray:
    k = int(np.clip(k, 2, 8))
    x = gray_u8.astype(np.float32) / 255.0
    mu = cv2.blur(x, (9, 9))
    mu2 = cv2.blur(x * x, (9, 9))
    var = np.clip(mu2 - mu * mu, 0.0, 1.0)

    feats = np.concatenate([x.reshape(-1, 1), var.reshape(-1, 1)], axis=1).astype(np.float32)
    m = (mask_u8.reshape(-1, 1) > 0).astype(np.float32)
    feats = feats * (0.2 + 0.8 * m)

    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6
    Z = (feats - mean) / std

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()

    centers_denorm = centers * std + mean
    intensity_centers = centers_denorm[:, 0]
    order = np.argsort(intensity_centers)
    inv = np.zeros_like(order)
    inv[order] = np.arange(k)
    labels_sorted = inv[labels]

    palette = np.array([
        [40, 60, 200],
        [60, 200, 200],
        [220, 220, 40],
        [60, 220, 60],
        [200, 60, 200],
        [255, 150, 60],
        [220, 60, 30],
        [240, 240, 240],
    ], dtype=np.uint8)
    rgb = palette[:k][labels_sorted].reshape(gray_u8.shape[0], gray_u8.shape[1], 3)
    return rgb


def _smooth_1d(arr: np.ndarray, ksize: int = 21) -> np.ndarray:
    ksize = int(np.clip(ksize, 3, 61))
    if ksize % 2 == 0:
        ksize += 1
    img = arr.astype(np.float32).reshape(256, 1)
    sm = cv2.GaussianBlur(img, (1, ksize), 0)
    return sm.reshape(256)


def color_transfer_reinhard_lut(ref_bgr_u8: np.ndarray, gray_u8: np.ndarray) -> np.ndarray:
    ref_lab = cv2.cvtColor(ref_bgr_u8, cv2.COLOR_BGR2LAB)
    Lr = ref_lab[:, :, 0].reshape(-1).astype(np.uint8)
    ar = ref_lab[:, :, 1].reshape(-1).astype(np.float32)
    br = ref_lab[:, :, 2].reshape(-1).astype(np.float32)

    counts = np.bincount(Lr, minlength=256).astype(np.float32)
    sum_a = np.bincount(Lr, weights=ar, minlength=256).astype(np.float32)
    sum_b = np.bincount(Lr, weights=br, minlength=256).astype(np.float32)

    mean_a = np.full(256, 128.0, dtype=np.float32)
    mean_b = np.full(256, 128.0, dtype=np.float32)
    nz = counts > 0
    mean_a[nz] = sum_a[nz] / counts[nz]
    mean_b[nz] = sum_b[nz] / counts[nz]

    mean_a = _smooth_1d(mean_a, 25)
    mean_b = _smooth_1d(mean_b, 25)

    lab = np.zeros((*gray_u8.shape, 3), dtype=np.uint8)
    lab[:, :, 0] = gray_u8
    lab[:, :, 1] = np.clip(mean_a[gray_u8], 0, 255).astype(np.uint8)
    lab[:, :, 2] = np.clip(mean_b[gray_u8], 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


MethodName = Literal[
    "LUT (Gradient)",
    "LUT (TURBO)",
    "LUT (VIRIDIS)",
    "LUT (INFERNO)",
    "Quantile Bands (Thyroid)",
    "KMeans Texture (Thyroid)",
    "Reinhard Transfer (Ref)",
]

METHOD_CHOICES = [
    "LUT (Gradient)",
    "LUT (TURBO)",
    "LUT (VIRIDIS)",
    "LUT (INFERNO)",
    "Quantile Bands (Thyroid)",
    "KMeans Texture (Thyroid)",
    "Reinhard Transfer (Ref)",
]


def colorize_thyroid_core(
    img: np.ndarray,
    method: MethodName,
    saturation: float,
    preprocess: bool,
    despeckle_strength: int,
    do_clahe: bool,
    clahe_clip: float,
    kmeans_k: int,
    ref_image: Optional[np.ndarray],
) -> np.ndarray:
    gray_u8 = to_gray_uint8(img)
    if gray_u8 is None:
        raise ValueError("Invalid input image")

    if preprocess:
        g, mask = preprocess_thyroid_ultrasound(
            gray_u8,
            mask_thresh=8,
            despeckle_strength=despeckle_strength,
            do_clahe=do_clahe,
            clahe_clip=clahe_clip,
            clahe_tile=8,
        )
    else:
        mask = make_ultrasound_scan_mask(gray_u8, thresh=8)
        g = gray_u8.copy()
        g[mask == 0] = 0

    if method == "LUT (Gradient)":
        out = apply_lut_rgb(g, LUT_GRADIENT)
    elif method.startswith("LUT ("):
        cmap = method.replace("LUT (", "").replace(")", "").strip()
        out = apply_opencv_colormap_rgb(g, cmap)
    elif method == "Quantile Bands (Thyroid)":
        out = thyroid_quantile_band_colorization(g, mask)
    elif method == "KMeans Texture (Thyroid)":
        out = kmeans_texture_colorization(g, mask, k=kmeans_k)
    elif method == "Reinhard Transfer (Ref)":
        ref = ensure_rgb_uint8(ref_image)
        if ref is None:
            out = apply_opencv_colormap_rgb(g, "TURBO")
        else:
            ref_bgr = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)
            if not ref_has_chroma(ref_bgr):
                out = apply_opencv_colormap_rgb(g, "TURBO")
            else:
                bgr = color_transfer_reinhard_lut(ref_bgr, g)
                out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

    out = apply_mask_rgb(out, mask)
    out = adjust_saturation(out, saturation)
    return out.astype(np.uint8)


def ui_run(inp, method, saturation, preprocess, despeckle_strength, do_clahe, clahe_clip, kmeans_k, ref):
    out = colorize_thyroid_core(
        img=inp,
        method=method,
        saturation=float(saturation),
        preprocess=bool(preprocess),
        despeckle_strength=int(despeckle_strength),
        do_clahe=bool(do_clahe),
        clahe_clip=float(clahe_clip),
        kmeans_k=int(kmeans_k),
        ref_image=ref,
    )
    return out


with gr.Blocks() as demo:
    gr.Markdown("## Thyroid US Colorization (Minimal UI)")

    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="Input B&W", type="numpy")
            method = gr.Dropdown(choices=METHOD_CHOICES, value="LUT (TURBO)", label="Method")
            saturation = gr.Slider(0, 2, value=1.0, step=0.05, label="Saturation")

            preprocess = gr.Checkbox(value=True, label="Preprocess (mask + despeckle + optional CLAHE)")
            despeckle_strength = gr.Slider(0, 20, value=7, step=1, label="Despeckle strength")
            do_clahe = gr.Checkbox(value=True, label="CLAHE")
            clahe_clip = gr.Slider(0.5, 6.0, value=2.0, step=0.1, label="CLAHE clip")

            kmeans_k = gr.Slider(2, 8, value=4, step=1, label="KMeans K (only for KMeans method)")
            ref = gr.Image(label="Reference (only for Reinhard)", type="numpy")
            btn = gr.Button("Apply")

        with gr.Column():
            out = gr.Image(label="Output", type="numpy")

    btn.click(ui_run, inputs=[inp, method, saturation, preprocess, despeckle_strength, do_clahe, clahe_clip, kmeans_k, ref], outputs=[out])


if __name__ == "__main__":
    demo.launch()