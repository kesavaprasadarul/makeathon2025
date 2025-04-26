# app.py  ──────────────────────────────────────────────────────────────
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import base64, json, io, cv2, numpy as np, torch
from pathlib import Path
from typing import List, Tuple
import hydra
from sam2.build_sam import build_sam2              # base model loader
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sys

sys.path.append(r"C:\Repos\makeathon2025\Kesava\repos\sam2")

# ──────────────────────────────────────────────────────────────────────
# Helper utilities
# ---------------------------------------------------------------------
def ndarray_to_png_b64(mask: np.ndarray) -> str:
    """Binary mask (H×W, uint8 0/255) ➜ base64-encoded PNG data-URI."""
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf).decode()

# ──────────────────────────────────────────────────────────────────────
# Heavy model initialisation (only once per container)
# ---------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM2_PREDICTOR = SAM2ImagePredictor(
    build_sam2(
        r"sam2_hiera_l.yaml",
        r"../checkpoints/sam2.1_hiera_large.pt",
        device=DEVICE,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
)

# ──────────────────────────────────────────────────────────────────────
# FastAPI
# ---------------------------------------------------------------------
app = FastAPI(
    title="Point-based SAM-2 segmentation API",
    description=(
        "Upload an image and supply point prompts to receive binary "
        "masks. Points are given in absolute image coordinates "
        "and must be accompanied by *point_labels* (1=foreground, "
        "0=background)."
    ),
)

@app.post("/segment")
async def segment(
    image: UploadFile = File(..., description="RGB/BGR image file"),
    points: str = Form(..., description='JSON list of [x, y] pairs'),
    point_labels: str = Form(..., description='JSON list of 1/0 labels'),
    multimask_output: bool = Form(False, description="If true, return 3 masks"),
):
    """
    Example curl:

    curl -F image=@myphoto.jpg \\
         -F points="[[420,200],[130,580]]" \\
         -F point_labels="[1,0]" \\
         http://localhost:8000/segment
    """
    # 1) read & decode image ------------------------------------------
    img_bytes = await image.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, "Could not decode image")

    # 2) parse point data ---------------------------------------------
    try:
        pts = np.array(json.loads(points), dtype=np.float32)   # (N,2)
        lbs = np.array(json.loads(point_labels), dtype=np.int64)  # (N,)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] != lbs.shape[0]:
            raise ValueError
    except Exception:
        raise HTTPException(
            400,
            "Invalid points / point_labels; provide JSON lists of equal length "
            "([[x,y],...], [1,0,...])",
        )

    # 3) run SAM-2 -----------------------------------------------------
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    SAM2_PREDICTOR.set_image(img_rgb)

    masks_np, scores, _ = SAM2_PREDICTOR.predict(
        point_coords=pts,
        point_labels=lbs,
        multimask_output=multimask_output,
        normalize_coords=True,
    )

    # 4) serialise masks ----------------------------------------------
    masks_png = [ndarray_to_png_b64(m[0] if m.ndim == 3 else m) for m in masks_np]

    return JSONResponse(
        {
            "masks_png":  masks_png,
            "scores":     scores.tolist(),
        }
    )

# ──────────────────────────────────────────────────────────────────────
# Uvicorn entry-point helper (optional)
# ---------------------------------------------------------------------
if _name_ == "_main_":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)