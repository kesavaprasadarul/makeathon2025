import os
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import cv2
import base64
from io import BytesIO
from typing import List
from datetime import datetime
import uuid

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = FastAPI()

# Configuration
MASK_SAVE_DIR = "generated_masks"
os.makedirs(MASK_SAVE_DIR, exist_ok=True)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize SAM2 model
sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# Global variables
current_image = None
points = []
labels = []

class ClickData(BaseModel):
    x: float
    y: float
    label: int = 1  # 1=foreground, 0=background

class ImageData(BaseModel):
    image_base64: str

def save_masks(masks: np.ndarray, scores: np.ndarray):
    """Save masks in multiple formats with timestamp and unique ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    base_filename = f"mask_{timestamp}_{unique_id}"
    
    # Create subdirectories for each format
    formats = ["torch", "numpy", "images"]
    for fmt in formats:
        os.makedirs(os.path.join(MASK_SAVE_DIR, fmt), exist_ok=True)
    
    saved_paths = []
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Save as PyTorch tensor
        torch_path = os.path.join(MASK_SAVE_DIR, "torch", f"{base_filename}_{i}_score_{score:.3f}.pt")
        torch.save(torch.from_numpy(mask), torch_path)
        
        # Save as numpy array
        numpy_path = os.path.join(MASK_SAVE_DIR, "numpy", f"{base_filename}_{i}_score_{score:.3f}.npy")
        np.save(numpy_path, mask)
        
        # Save as PNG image
        img_path = os.path.join(MASK_SAVE_DIR, "images", f"{base_filename}_{i}_score_{score:.3f}.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(img_path)
        
        saved_paths.append({
            "torch": torch_path,
            "numpy": numpy_path,
            "image": img_path,
            "score": float(score)
        })
    
    return saved_paths

@app.post("/set_image")
async def set_image(data: ImageData):
    """Set the base64 encoded image for segmentation"""
    global current_image, predictor, points, labels
    try:
        # Handle both data:image/png;base64,XXX and raw base64 strings
        if data.image_base64.startswith('data:image'):
            image_data = base64.b64decode(data.image_base64.split(",")[1])
        else:
            image_data = base64.b64decode(data.image_base64)
            
        image = Image.open(BytesIO(image_data)).convert("RGB")
        current_image = np.array(image)
        predictor.set_image(current_image)
        points = []
        labels = []
        return {"message": "Image set successfully", "image_size": current_image.shape[:2]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

@app.post("/add_point")
async def add_point(click: ClickData):
    """Add a point and return the generated masks"""
    global points, labels, current_image
    try:
        points.append([click.x, click.y])
        labels.append(click.label)
        
        masks, scores, _ = predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels),
            multimask_output=True,
        )
        
        # Save masks in all formats
        saved_paths = save_masks(masks, scores)
        
        # Convert masks to base64 for frontend
        mask_images = []
        for mask in masks:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            buffered = BytesIO()
            mask_img.save(buffered, format="PNG")
            mask_images.append(base64.b64encode(buffered.getvalue()).decode())
        
        return {
            "masks": mask_images,
            "scores": scores.tolist(),
            "points": points,
            "image_size": current_image.shape[:2],
            "saved_paths": saved_paths  # Return paths for debugging
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reset")
async def reset():
    """Reset all points"""
    global points, labels
    points = []
    labels = []
    return {"message": "Points reset"}

@app.get("/debug/masks")
async def debug_masks():
    """Debug endpoint to list saved masks"""
    try:
        mask_files = {}
        for root, dirs, files in os.walk(MASK_SAVE_DIR):
            format_name = os.path.basename(root)
            if format_name in ["torch", "numpy", "images"]:
                mask_files[format_name] = files
        return mask_files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)