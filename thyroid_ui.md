# Thyroid US Colorization UI (Gradio)

This is the little Gradio app I use to quickly preview my thyroid ultrasound colorization methods on single images.

## What I use it for
- Upload a B&W thyroid US image and instantly see a colorized result
- Flip between methods (LUTs, quantile bands, KMeans texture, Reinhard transfer)
- Toggle preprocessing (scan mask + despeckle + optional CLAHE)
- Adjust saturation to sanity-check how aggressive the colors feel


## Run it
From the folder where `thyroid_ui.py` lives:

```bash
pip install gradio opencv-python numpy
python thyroid_ui.py
```

It’ll print a local URL (usually `http://127.0.0.1:7860`) — open it in your browser.

## Notes
- **Reinhard Transfer** uses the “Reference” image input. If the reference has no real chroma, it falls back to TURBO.
- Inputs can be JPG/PNG. The app converts to grayscale internally and masks outside the scan region.
