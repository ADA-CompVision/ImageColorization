# Thyroid batch generator (TN5000 — Original only)

This is the small script I use to generate dataset of colorized outputs from **TN5000 original** thyroid ultrasound images.

## What it does
- Reads images from **TN5000 playground/Original US Images from TN5000/**
- Ignores **Segmented US Images from TN5000** completely
- Generates **54 outputs per image** (methods × preprocessing × saturation × k for KMeans)
- Saves everything under an output folder you choose

## Input layout (TN5000)
```
TN5000 playground/
  Original US Images from TN5000/
    benign/     000001.jpg ...
    malignant/  000001.jpg ...
```

## Strict ID + type in filename
Original filenames are digits-only, like `000001.jpg`.

Output filenames look like:
```
ID_type_method_preproc_saturation[_k].jpg
```

- `ID` = original filename stem (e.g., `000001`)
- `type` = `0` for benign, `1` for malignant (based on the folder)
- `method` codes:
  - 01 Gradient, 02 TURBO, 03 VIRIDIS, 04 INFERNO
  - 05 Quantile Bands, 06 KMeans Texture, 07 Reinhard (falls back to TURBO in batch)
- `preproc` = 0/1
- `saturation` = 1 / 1,5 / 2
- `_k` only for KMeans (2/5/8)

No region-based method exists in this script.

## Run (full batch)
```
python thyroid_batch.py generate-tn5000 \
  --playground_root "$HOME/Downloads/TN5000 playground" \
  --output_root "$HOME/Downloads/thyroid_img_result"
```

Outputs go to:
```
$HOME/Downloads/thyroid_img_result/original/
```

Quick count check:
```
ls -1 "$HOME/Downloads/thyroid_img_result/original" | wc -l
```
