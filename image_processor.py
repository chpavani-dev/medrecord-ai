import cv2
import numpy as np

def enhance_image(img_array):
    """
    Light enhancement only — preserve text quality for OCR.
    """
    # Convert to BGR if needed
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Resize if too small
    h, w = img_array.shape[:2]
    if w < 1500:
        scale  = 1500 / w
        img_array = cv2.resize(img_array, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Light denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=5)

    # Contrast enhancement (gentle)
    clahe     = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    contrasted = clahe.apply(denoised)

    return contrasted


def get_image_quality_score(img_array):
    """Score image quality 0-100"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {
        "score":   min(blur_score / 500 * 100, 100),
        "is_good": blur_score > 100
    }