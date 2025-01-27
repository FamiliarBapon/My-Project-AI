from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline
import uuid
import os
import logging
from typing import Optional
import numpy as np
import cv2
from PIL import Image
import boto3
from datetime import datetime
from dotenv import load_dotenv

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from pydantic import BaseModel
except ImportError as e:
    print(f"FastAPI related import error: {e}")
    print("Please install required packages: pip install fastapi python-multipart")
    raise

try:
    import torch
    from diffusers import StableDiffusionPipeline
    from transformers import pipeline
except ImportError as e:
    print(f"AI model related import error: {e}")
    print("Please install required packages: pip install torch diffusers transformers")
    raise

try:
    import numpy as np
    import cv2
    from PIL import Image
except ImportError as e:
    print(f"Image processing related import error: {e}")
    print("Please install required packages: pip install numpy opencv-python Pillow")
    raise

try:
    import boto3
except ImportError as e:
    print(f"AWS related import error: {e}")
    print("Please install required package: pip install boto3")
    raise

import uuid
import os
import logging
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Check CUDA availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Verify required environment variables
required_env_vars = ["AWS_ACCESS_KEY", "AWS_SECRET_KEY", "S3_BUCKET"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Create necessary directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
for directory in [UPLOAD_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize AWS client with error handling
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )
except Exception as e:
    logger.error(f"Failed to initialize AWS client: {e}")
    raise

# Verify model availability
MODEL_PATH = os.getenv("MODEL_PATH", "runwayml/stable-diffusion-v1-5")
try:
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Using CPU mode (slower performance)")
except Exception as e:
    logger.error(f"Error checking CUDA availability: {e}")
    raise

# Memory management for CUDA
if DEVICE == "cuda":
    try:
        torch.cuda.empty_cache()
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except Exception as e:
        logger.error(f"Error managing CUDA memory: {e}")
        raise

def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """Clean up old temporary files"""
    try:
        current_time = datetime.now()
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            file_age = datetime.fromtimestamp(os.path.getctime(filepath))
            if (current_time - file_age).total_seconds() > max_age_hours * 3600:
                os.remove(filepath)
                logger.info(f"Cleaned up old file: {filepath}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Run cleanup on startup
for directory in [UPLOAD_DIR, OUTPUT_DIR]:
    cleanup_old_files(directory)

# Version check for critical dependencies
def check_versions():
    try:
        from importlib.metadata import version as get_version
        
        required_versions = {
            'torch': '1.9.0',
            'fastapi': '0.68.0',
            'diffusers': '0.12.0',
            'transformers': '4.21.0'
        }
        
        for package, min_version in required_versions.items():
            try:
                installed_version = get_version(package)
                if installed_version < min_version:
                    logger.warning(f"{package} version {installed_version} is lower than recommended version {min_version}")
            except Exception:
                logger.error(f"Package {package} not found")
    except Exception as e:
        logger.error(f"Error checking package versions: {e}")

check_versions()

# Export constants for use in other parts of the application
__all__ = [
    'DEVICE',
    'UPLOAD_DIR',
    'OUTPUT_DIR',
    'MODEL_PATH',
    's3_client',
    'logger'
]

# Initialize FastAPI app
app = FastAPI(title="AI Generation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables (store these securely)
MODEL_PATH = "runwayml/stable-diffusion-v1-5"  # or your preferred model
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Request/Response Models
class GenerationRequest(BaseModel):
    prompt: str
    style: Optional[str] = "realistic"
    ratio: Optional[str] = "1:1"
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    num_images: Optional[int] = 1

class GenerationResponse(BaseModel):
    status: str
    urls: list
    generation_id: str

# Initialize AI Models
class AIGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_model = StableDiffusionPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # Add safety checker
        self.safety_checker = pipeline("image-classification", model="openai/clip-vit-base-patch32")
        
        # Create output directory
        os.makedirs("outputs", exist_ok=True)

    def generate_image(self, prompt: str, style: str, ratio: str, 
                      negative_prompt: str = None, seed: int = None,
                      num_images: int = 1) -> list:
        try:
            # Apply style modifiers
            styled_prompt = self._apply_style(prompt, style)
            
            # Set dimensions based on ratio
            width, height = self._get_dimensions(ratio)
            
            # Set random seed if not provided
            if seed is None:
                seed = torch.randint(0, 1000000, (1,)).item()
            
            # Generate images
            generator = torch.Generator(device=self.device).manual_seed(seed)
            images = self.image_model(
                prompt=styled_prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                generator=generator,
                width=width,
                height=height
            ).images
            
            # Save and upload images
            urls = []
            for idx, image in enumerate(images):
                # Check image safety
                if self._is_safe_image(image):
                    url = self._save_and_upload_image(image, f"generation_{uuid.uuid4()}_{idx}.png")
                    urls.append(url)
                else:
                    logger.warning(f"Image {idx} failed safety check")
            
            return urls
        
        except Exception as e:
            logger.error(f"Image generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def _apply_style(self, prompt: str, style: str) -> str:
        style_modifiers = {
            "realistic": "highly detailed, photorealistic, 8k resolution",
            "artistic": "artistic style, vibrant colors, painterly effect",
            "cartoon": "cartoon style, cel shaded, vibrant",
            "3d": "3D rendered, octane render, highly detailed"
        }
        return f"{prompt}, {style_modifiers.get(style, '')}"

    def _get_dimensions(self, ratio: str) -> tuple:
        base_size = 512
        ratio_map = {
            "1:1": (base_size, base_size),
            "16:9": (int(base_size * 16/9), base_size),
            "4:3": (int(base_size * 4/3), base_size),
            "9:16": (base_size, int(base_size * 16/9))
        }
        return ratio_map.get(ratio, (base_size, base_size))

    def _is_safe_image(self, image) -> bool:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Basic safety checks (example)
        try:
            # Convert to RGB if necessary
            if len(img_array.shape) == 2:
                return False  # Reject grayscale images
            
            # Check for extreme brightness/darkness
            brightness = np.mean(img_array)
            if brightness < 20 or brightness > 235:
                return False
            
            # Use CLIP model for content classification
            results = self.safety_checker(image)
            unsafe_labels = ["nsfw", "violence", "gore"]
            
            for result in results:
                if result["label"].lower() in unsafe_labels and result["score"] > 0.5:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Safety check error: {str(e)}")
            return False

    def _save_and_upload_image(self, image: Image, filename: str) -> str:
        try:
            # Save locally first
            local_path = os.path.join("outputs", filename)
            image.save(local_path)
            
            # Upload to S3
            s3_client.upload_file(
                local_path,
                S3_BUCKET,
                f"generations/{filename}",
                ExtraArgs={'ACL': 'public-read'}
            )
            
            # Generate URL
            url = f"https://{S3_BUCKET}.s3.amazonaws.com/generations/{filename}"
            
            # Clean up local file
            os.remove(local_path)
            
            return url
            
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            raise e

# Initialize AI Generator
generator = AIGenerator()

# API Endpoints
@app.post("/generate/image", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    try:
        # Generate images
        urls = generator.generate_image(
            prompt=request.prompt,
            style=request.style,
            ratio=request.ratio,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            num_images=request.num_images
        )
        
        # Generate unique ID for this generation
        generation_id = str(uuid.uuid4())
        
        return GenerationResponse(
            status="success",
            urls=urls,
            generation_id=generation_id
        )
        
    except Exception as e:
        logger.error(f"Generation endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": str(exc.detail)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"}
    )