#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from pathlib import Path
from typing import Optional, Dict, Tuple, Literal, List

import cv2
import numpy as np


# ============================================================
# STRICT ID extraction (digits-only filename, no fallbacks)
# ============================================================

def extract_id_from_filename(fp: Path, pad: int = 6) -> str:
    """
    STRICT:
      - fp.stem must be digits only (e.g., "000001")
      - returns zero-padded to 'pad' (doesn't change if already >= pad digits)
    """
    s = fp.stem
    if not s.isdigit():
        raise ValueError(
            f"Invalid filename for strict ID extraction: {fp.name}\n"
            "Expected digits-only filenames like: 000001.jpg"
        )
    return s.zfill(pad)


# ============================================================
# Helpers: dtype/format
# ============================================================

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


# ============================================================
# Thyroid preprocessing (scan mask + despeckle + CLAHE)
# ============================================================

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


# ============================================================
# Colorization methods (NO region-based)
# ============================================================

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

    return palette[:k][labels_sorted].reshape(gray_u8.shape[0], gray_u8.shape[1], 3)


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


# ============================================================
# Batch config
# ============================================================

MethodName = Literal[
    "LUT (Gradient)",
    "LUT (TURBO)",
    "LUT (VIRIDIS)",
    "LUT (INFERNO)",
    "Quantile Bands (Thyroid)",
    "KMeans Texture (Thyroid)",
    "Reinhard Transfer (Ref)",
]

METHOD_CHOICES: List[str] = [
    "LUT (Gradient)",
    "LUT (TURBO)",
    "LUT (VIRIDIS)",
    "LUT (INFERNO)",
    "Quantile Bands (Thyroid)",
    "KMeans Texture (Thyroid)",
    "Reinhard Transfer (Ref)",
]

METHOD_CODE: Dict[str, str] = {
    "LUT (Gradient)": "01",
    "LUT (TURBO)": "02",
    "LUT (VIRIDIS)": "03",
    "LUT (INFERNO)": "04",
    "Quantile Bands (Thyroid)": "05",
    "KMeans Texture (Thyroid)": "06",
    "Reinhard Transfer (Ref)": "07",
}

SAT_TOKENS = ["1", "1,5", "2"]  # keep EXACT tokens in filenames
KMEANS_K_CHOICES = [2, 5, 8]


def sat_token_to_float(tok: str) -> float:
    return float(tok.replace(",", "."))


# ============================================================
# IO
# ============================================================

def read_image_rgb_or_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_jpg_rgb(path: Path, rgb_u8: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_u8 = ensure_rgb_uint8(rgb_u8)
    if rgb_u8 is None:
        raise ValueError("Tried to save None image")
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 50, 100))])
    if not ok:
        raise ValueError(f"Failed to write: {path}")


# ============================================================
# Core colorize used in batch
# ============================================================

def colorize_core(
    img: np.ndarray,
    method: MethodName,
    saturation: float,
    preprocess: bool,
    kmeans_k: int,
    ref_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    gray_u8 = to_gray_uint8(img)
    if gray_u8 is None:
        raise ValueError("Invalid input image")

    if preprocess:
        g, mask = preprocess_thyroid_ultrasound(
            gray_u8,
            mask_thresh=8,
            despeckle_strength=7,
            do_clahe=True,
            clahe_clip=2.0,
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


def build_combos() -> list:
    combos = []
    for preprocess_flag in [False, True]:
        pre_bit = 1 if preprocess_flag else 0
        for method in METHOD_CHOICES:
            mcode = METHOD_CODE[method]
            if method == "KMeans Texture (Thyroid)":
                for sat_tok in SAT_TOKENS:
                    for kk in KMEANS_K_CHOICES:
                        combos.append((method, mcode, pre_bit, sat_tok, int(kk)))
            else:
                for sat_tok in SAT_TOKENS:
                    combos.append((method, mcode, pre_bit, sat_tok, None))
    return combos


# ============================================================
# Generators
# ============================================================

def _assert_unique_ids(items: List[Path]) -> None:
    seen = set()
    for fp in items:
        id_tok = extract_id_from_filename(fp)
        if id_tok in seen:
            raise ValueError(f"Duplicate ID detected: {id_tok} (example file: {fp})")
        seen.add(id_tok)


def generate_generic(
    input_root: Path,
    output_root: Path,
    seed: int,
    shuffle: bool,
    jpg_quality: int,
    limit_images: int,
) -> None:
    # expects: input_root/{benign,malignant}/...
    class_map = {"benign": 0, "malignant": 1}
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    items: List[Tuple[Path, int]] = []
    for dname, cls in class_map.items():
        d = input_root / dname
        if not d.exists():
            continue
        for fp in sorted(d.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in exts:
                items.append((fp, cls))

    if not items:
        raise SystemExit(f"No images found under {input_root} with subfolders {list(class_map.keys())}")

    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(items)

    if limit_images and limit_images > 0:
        items = items[:limit_images]

    # strict ID + uniqueness checks
    _assert_unique_ids([fp for fp, _ in items])

    combos = build_combos()
    output_root.mkdir(parents=True, exist_ok=True)

    for fp, cls in items:
        img = read_image_rgb_or_gray(fp)
        id_tok = extract_id_from_filename(fp)

        for method, mcode, pre_bit, sat_tok, kk in combos:
            sat_val = sat_token_to_float(sat_tok)
            out = colorize_core(
                img=img,
                method=method,  # type: ignore
                saturation=sat_val,
                preprocess=bool(pre_bit),
                kmeans_k=int(kk) if kk is not None else 4,
                ref_image=None,
            )

            # ID then TYPE (0 benign / 1 malignant) immediately after ID
            parts = [id_tok, str(cls), str(mcode), str(pre_bit), sat_tok]
            if method == "KMeans Texture (Thyroid)" and kk is not None:
                parts.append(str(int(kk)))
            fname = "_".join(parts) + ".jpg"
            save_jpg_rgb(output_root / fname, out, quality=jpg_quality)

    print(f"Done. Output: {output_root}")


def generate_tn5000(
    playground_root: Path,
    output_root: Path,
    seed: int,
    shuffle: bool,
    jpg_quality: int,
    limit_images: int,
) -> None:
    """
    ONLY reads:
      <root>/Original US Images from TN5000/{benign,malignant}/*.jpg

    Writes:
      <output_root>/original/*.jpg
    """
    pg = playground_root

    # resolve common nesting: "TN5000 playground/TN5000 playground/..."
    candidate_same_name = pg / pg.name
    if candidate_same_name.exists() and (candidate_same_name / "Original US Images from TN5000").exists():
        pg = candidate_same_name

    orig = pg / "Original US Images from TN5000"
    if not orig.exists():
        raise SystemExit(f"Could not find: {orig}")

    combos = build_combos()
    rng = random.Random(seed)

    step_out = output_root / "original"
    step_out.mkdir(parents=True, exist_ok=True)

    items: List[Tuple[Path, int]] = []
    for cls_name in ["benign", "malignant"]:
        d = orig / cls_name
        if not d.exists():
            continue
        label = 0 if cls_name == "benign" else 1
        for fp in sorted(d.glob("*.jpg")):
            items.append((fp, label))

    if shuffle:
        rng.shuffle(items)

    if limit_images and limit_images > 0:
        items = items[:limit_images]

    # strict ID + uniqueness checks
    _assert_unique_ids([fp for fp, _ in items])

    for fp, cls in items:
        img = read_image_rgb_or_gray(fp)
        id_tok = extract_id_from_filename(fp)

        for method, mcode, pre_bit, sat_tok, kk in combos:
            sat_val = sat_token_to_float(sat_tok)
            out = colorize_core(
                img=img,
                method=method,  # type: ignore
                saturation=sat_val,
                preprocess=bool(pre_bit),
                kmeans_k=int(kk) if kk is not None else 4,
                ref_image=None,
            )

            # ID then TYPE (0 benign / 1 malignant) immediately after ID
            parts = [id_tok, str(cls), str(mcode), str(pre_bit), sat_tok]
            if method == "KMeans Texture (Thyroid)" and kk is not None:
                parts.append(str(int(kk)))
            fname = "_".join(parts) + ".jpg"
            save_jpg_rgb(step_out / fname, out, quality=jpg_quality)

    print(f"TN5000 original done -> {step_out} (images processed: {len(items)})")


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser("Minimal batch generators (NO region-based)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("generate-generic", help="input_root/{benign,malignant} -> output_root")
    p1.add_argument("--input_root", required=True)
    p1.add_argument("--output_root", required=True)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--no_shuffle", action="store_true")
    p1.add_argument("--jpg_quality", type=int, default=95)
    p1.add_argument("--limit_images", type=int, default=0, help="Process only first N images (0=all)")

    p2 = sub.add_parser("generate-tn5000", help="TN5000 playground ORIGINAL only -> output_root/original")
    p2.add_argument("--playground_root", required=True)
    p2.add_argument("--output_root", required=True)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--no_shuffle", action="store_true")
    p2.add_argument("--jpg_quality", type=int, default=95)
    p2.add_argument("--limit_images", type=int, default=0, help="Process only first N images (0=all)")

    args = ap.parse_args()

    if args.cmd == "generate-generic":
        generate_generic(
            input_root=Path(args.input_root).expanduser(),
            output_root=Path(args.output_root).expanduser(),
            seed=int(args.seed),
            shuffle=not bool(args.no_shuffle),
            jpg_quality=int(args.jpg_quality),
            limit_images=int(args.limit_images),
        )
    else:
        generate_tn5000(
            playground_root=Path(args.playground_root).expanduser(),
            output_root=Path(args.output_root).expanduser(),
            seed=int(args.seed),
            shuffle=not bool(args.no_shuffle),
            jpg_quality=int(args.jpg_quality),
            limit_images=int(args.limit_images),
        )


if __name__ == "__main__":
    main()