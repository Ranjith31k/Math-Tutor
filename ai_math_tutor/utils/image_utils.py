from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import io

def load_image(pil_image):
    """Ensure PIL.Image instance and return OpenCV image (BGR)."""
    if not isinstance(pil_image, Image.Image):
        pil_image = Image.open(pil_image)
    rgb = pil_image.convert("RGB")
    arr = np.array(rgb)
    # Convert RGB to BGR for OpenCV
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def preprocess_for_ocr(pil_image, max_dim=1600):
    """Resize, denoise, and enhance contrast for OCR."""
    try:
        # Resize while keeping aspect ratio
        w, h = pil_image.size
        scale = min(1.0, float(max_dim) / max(w, h))
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)

        # Convert to grayscale and apply unsharp mask
        gray = pil_image.convert("L")
        enhanced = ImageOps.autocontrast(gray)
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

        return enhanced
    except Exception:
        return pil_image.convert("L")

def to_bytes(pil_image, fmt="PNG"):
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()
