from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import cv2
import base64
from typing import List, Optional
import tempfile
import os
import time
import logging
import sys
from mnemonic import Mnemonic

from utils import (
    load_detection_model,
    load_recognition_model,
    load_binary_encoder,
    process_batch,
    extract_frames_from_video
)

# Set up more verbose logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI with custom configurations
app = FastAPI(
    title="Face Binary API"
)

# Add middleware with explicit limits
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
detection_model = None
recognition_model = None
binary_model = None
mnemo = Mnemonic("english")

@app.on_event("startup")
async def startup_event():
    global detection_model, recognition_model, binary_model

    try:
        logger.info("Loading detection model...")
        detection_model = load_detection_model()

        logger.info("Loading recognition model...")
        recognition_model = load_recognition_model()

        logger.info("Loading binary encoder model...")
        binary_model = load_binary_encoder()

        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

# Define request model with explicit size limits
class ImageRequest(BaseModel):
    images: List[str] = Field(..., description="List of base64 encoded images")

class ProcessResponse(BaseModel):
    binary_codes: List[List[int]]
    settled_vector: List[int]
    mnemonic: str
    processing_time: float

def get_settled_vector(binary_codes):
    """
    Calculate the most common bit value for each position across all binary codes.
    Returns a single binary vector of length 16.
    """
    if not binary_codes:
        return [0] * 16  # Default to all zeros if no codes

    # Convert to numpy array for easier processing
    codes_array = np.array(binary_codes)

    # Get the most common value (0 or 1) for each of the 16 positions
    settled_vector = []
    for i in range(codes_array.shape[1]):
        # Count occurrences of 1s in this position
        count_ones = np.sum(codes_array[:, i])
        # If more than half are 1s, use 1, otherwise use 0
        settled_vector.append(1 if count_ones > len(binary_codes) / 2 else 0)

    return settled_vector

def generate_mnemonic(binary_vector):
    """
    Generate a 12-word BIP39 mnemonic from a 16-bit binary vector.
    """
    # Convert binary vector to bytes
    # We need at least 128 bits (16 bytes) for BIP39, so we'll repeat our 16 bits
    # to fill 128 bits (or 16 bytes)
    binary_str = ''.join(map(str, binary_vector))
    # Repeat the 16 bits to get at least 128 bits
    repeated_str = binary_str * 8  # 16 * 8 = 128 bits

    # Convert to bytes
    bytes_array = bytearray()
    for i in range(0, len(repeated_str), 8):
        byte = int(repeated_str[i:i+8], 2)
        bytes_array.append(byte)

    # Generate mnemonic from bytes
    return mnemo.to_mnemonic(bytes(bytes_array))

# Process images in batches to handle large requests
def process_images_in_batches(images, batch_size=8):
    all_binary_codes = []
    
    # Process in smaller batches to avoid memory issues
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(images) + batch_size - 1)//batch_size}")
        batch_codes = process_batch(batch, detection_model, recognition_model, binary_model)
        all_binary_codes.extend(batch_codes)
        # Free up memory
        batch = None
    
    return all_binary_codes

# Add a custom middleware to log raw request data before parsing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request received: {request.method} {request.url.path}")
    
    # Log request headers for debugging
    logger.info(f"Request headers: {request.headers}")
    
    # Process the request
    response = await call_next(request)
    
    logger.info(f"Response status: {response.status_code}")
    return response

@app.post("/process_images")
async def process_images(request: Request):
    logger.info("Starting to process images request")
    start_time = time.time()
    
    # Get raw request data
    try:
        # Read the raw request body
        body = await request.json()
        logger.info(f"Successfully parsed request JSON, contains {len(body.get('images', []))} images")
    except Exception as e:
        logger.error(f"Failed to parse request JSON: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"detail": f"Invalid JSON in request: {str(e)}"}
        )
    
    # Extract images from request
    image_data = body.get("images", [])
    if not image_data:
        logger.warning("No images found in request")
        return JSONResponse(
            content={
                "binary_codes": [],
                "settled_vector": [0] * 16,
                "mnemonic": "",
                "processing_time": time.time() - start_time
            }
        )
    
    # Log request size
    request_size_mb = sum(len(img_str) for img_str in image_data) / (1024 * 1024)
    logger.info(f"Processing {len(image_data)} images, total size: {request_size_mb:.2f} MB")
    
    # Process images
    images = []
    for i, img_str in enumerate(image_data):
        try:
            # Log progress occasionally
            if i % 10 == 0:
                logger.info(f"Decoding image {i+1}/{len(image_data)}")
                
            img_data = base64.b64decode(img_str)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is not None:
                images.append(img)
            else:
                logger.warning(f"Failed to decode image {i+1}")
        except Exception as e:
            logger.error(f"Error decoding image {i+1}: {str(e)}")
            continue
    
    if not images:
        logger.warning("No valid images could be decoded")
        return JSONResponse(
            content={
                "binary_codes": [],
                "settled_vector": [0] * 16,
                "mnemonic": "",
                "processing_time": time.time() - start_time
            }
        )
    
    logger.info(f"Successfully decoded {len(images)} images, starting processing")
    
    # Process images in smaller batches
    binary_codes = process_images_in_batches(images, batch_size=4)
    
    # Clean up to free memory
    images = None
    
    if not binary_codes:
        logger.warning("No binary codes were generated")
        return JSONResponse(
            content={
                "binary_codes": [],
                "settled_vector": [0] * 16,
                "mnemonic": "",
                "processing_time": time.time() - start_time
            }
        )
    
    # Get the settled vector (most common bit per position)
    settled_vector = get_settled_vector(binary_codes)
    
    # Generate mnemonic from the settled vector
    mnemonic = generate_mnemonic(settled_vector)
    
    processing_time = time.time() - start_time
    logger.info(f"Completed processing in {processing_time:.2f} seconds")
    
    return JSONResponse(
        content={
            "binary_codes": binary_codes,
            "settled_vector": settled_vector,
            "mnemonic": mnemonic,
            "processing_time": processing_time
        }
    )

@app.post("/process_video")
async def process_video(
    video: UploadFile = File(...),
    max_frames: Optional[int] = Form(None)
):
    logger.info(f"Starting to process video: {video.filename}")
    start_time = time.time()

    # Read video data in chunks to avoid memory issues
    chunks = []
    chunk_size = 1024 * 1024  # 1MB chunks
    
    try:
        while True:
            chunk = await video.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
    except Exception as e:
        logger.error(f"Error reading video file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error reading video file: {str(e)}")
    
    if not chunks:
        logger.warning("Empty video file received")
        raise HTTPException(status_code=400, detail="Empty video file")
    
    video_bytes = b''.join(chunks)
    logger.info(f"Read {len(video_bytes) / (1024 * 1024):.2f} MB of video data")

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_file.write(video_bytes)
        temp_path = temp_file.name
    
    # Free memory
    video_bytes = None
    chunks = None

    try:
        logger.info(f"Extracting frames from video at {temp_path}")
        frames = extract_frames_from_video(temp_path, max_frames)
        logger.info(f"Extracted {len(frames)} frames from video")
        
        if not frames:
            logger.warning("No frames could be extracted from the video")
            return JSONResponse(
                content={
                    "binary_codes": [],
                    "settled_vector": [0] * 16,
                    "mnemonic": "",
                    "processing_time": time.time() - start_time
                }
            )

        # Process frames in smaller batches
        binary_codes = process_images_in_batches(frames, batch_size=4)
        
        # Free memory
        frames = None
        
        if not binary_codes:
            logger.warning("No binary codes were generated from video frames")
            return JSONResponse(
                content={
                    "binary_codes": [],
                    "settled_vector": [0] * 16,
                    "mnemonic": "",
                    "processing_time": time.time() - start_time
                }
            )

        # Get the settled vector (most common bit per position)
        settled_vector = get_settled_vector(binary_codes)

        # Generate mnemonic from the settled vector
        mnemonic = generate_mnemonic(settled_vector)

        processing_time = time.time() - start_time
        logger.info(f"Completed video processing in {processing_time:.2f} seconds")
        
        return JSONResponse(
            content={
                "binary_codes": binary_codes,
                "settled_vector": settled_vector,
                "mnemonic": mnemonic,
                "processing_time": processing_time
            }
        )
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Removed temporary file: {temp_path}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded": {
        "detection": detection_model is not None,
        "recognition": recognition_model is not None,
        "binary_encoder": binary_model is not None
    }}

if __name__ == "__main__":
    import uvicorn
    
    # Run server with increased limits for large requests
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        limit_concurrency=10,
        timeout_keep_alive=300,
        timeout_graceful_shutdown=600,
        h11_max_incomplete_event_size=104857600  # 100MB
    )