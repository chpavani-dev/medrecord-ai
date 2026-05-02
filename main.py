"""
MedRecord OCR Microservice
FastAPI + Google Vision (REST) + EasyOCR fallback + Claude Haiku 4.5 drug extraction

Environment variables required on Railway:
- GOOGLE_VISION_KEY    : Google Cloud Vision API key (starts with AIzaSy...)
- ANTHROPIC_API_KEY    : Claude API key from console.anthropic.com
"""

import os
import base64
import json
import logging
from typing import Optional

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("medrecord-ocr")

# ------------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------------
GOOGLE_VISION_KEY = os.getenv("GOOGLE_VISION_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not GOOGLE_VISION_KEY:
    logger.warning("GOOGLE_VISION_KEY not set — Google Vision will be skipped")
if not ANTHROPIC_API_KEY:
    logger.warning("ANTHROPIC_API_KEY not set — Claude drug extraction will be skipped")

# ------------------------------------------------------------------
# EasyOCR — load lazily (heavy dependency)
# ------------------------------------------------------------------
_easyocr_reader = None

def get_easyocr_reader():
    """Lazy-load EasyOCR to avoid slow cold starts when not needed."""
    global _easyocr_reader
    if _easyocr_reader is None:
        logger.info("Loading OCR engine...")
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
        logger.info("OCR engine ready!")
    return _easyocr_reader

# ------------------------------------------------------------------
# FastAPI app setup
# ------------------------------------------------------------------
app = FastAPI(
    title="MedRecord OCR Service",
    description="OCR + drug extraction for lab reports and prescriptions",
    version="1.1.0",
)

# Allow Expo/React Native app to call this from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------
class OCRResponse(BaseModel):
    success: bool
    text: str
    engine: str  # "google_vision" or "easyocr"
    error: Optional[str] = None

class DrugExtractionResponse(BaseModel):
    success: bool
    text: str
    engine: str
    drugs: list
    error: Optional[str] = None

# ------------------------------------------------------------------
# OCR functions
# ------------------------------------------------------------------
def extract_text_google_vision(image_bytes: bytes) -> str:
    """
    Extract text using Google Vision REST API.
    Uses the API key from GOOGLE_VISION_KEY env variable.
    Raises exception on failure so caller can fall back to EasyOCR.
    """
    if not GOOGLE_VISION_KEY:
        raise ValueError("GOOGLE_VISION_KEY not configured")

    image_content = base64.b64encode(image_bytes).decode("utf-8")

    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_KEY}"
    payload = {
        "requests": [
            {
                "image": {"content": image_content},
                "features": [{"type": "TEXT_DETECTION", "maxResults": 1}],
            }
        ]
    }

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()

    # Defensive parsing — this is the bug that was logging "Google Vision failed: 'responses'"
    responses = result.get("responses", [])
    if not responses:
        logger.warning("Google Vision returned no 'responses' field. Raw: %s", result)
        return ""

    first = responses[0]

    # API may return an error inside the response object even with HTTP 200
    if "error" in first:
        err = first["error"]
        raise RuntimeError(f"Google Vision API error: {err.get('message', err)}")

    text_annotations = first.get("textAnnotations", [])
    if not text_annotations:
        return ""

    # The first annotation contains the full extracted text
    return text_annotations[0].get("description", "")


def extract_text_easyocr(image_bytes: bytes) -> str:
    """Fallback OCR using EasyOCR (slower, runs on CPU)."""
    import numpy as np
    import cv2

    reader = get_easyocr_reader()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")

    results = reader.readtext(img, detail=0, paragraph=True)
    return "\n".join(results)


def extract_text_with_fallback(image_bytes: bytes) -> tuple[str, str]:
    """
    Try Google Vision first, fall back to EasyOCR.
    Returns (extracted_text, engine_used).
    """
    # Try Google Vision
    try:
        text = extract_text_google_vision(image_bytes)
        if text and text.strip():
            logger.info("Google Vision succeeded (%d chars)", len(text))
            return text, "google_vision"
        logger.warning("Google Vision returned empty text — falling back to EasyOCR")
    except Exception as e:
        logger.warning("Google Vision failed: %s — falling back to EasyOCR", e)

    # Fall back to EasyOCR
    try:
        text = extract_text_easyocr(image_bytes)
        logger.info("EasyOCR succeeded (%d chars)", len(text))
        return text, "easyocr"
    except Exception as e:
        logger.error("EasyOCR also failed: %s", e)
        raise


# ------------------------------------------------------------------
# Claude — drug extraction
# ------------------------------------------------------------------
def extract_drugs_with_claude(ocr_text: str) -> list:
    """Use Claude Haiku 4.5 to extract structured drug info from OCR text."""
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — skipping drug extraction")
        return []

    if not ocr_text or not ocr_text.strip():
        return []

    prompt = f"""You are a medical assistant extracting medications from a prescription or hospital discharge summary.

OCR text from prescription:
---
{ocr_text}
---

Extract every medication mentioned. For each drug, return:
- name: generic or brand name as written
- dosage: e.g. "500mg", "10ml" (null if not specified)
- frequency: e.g. "twice daily", "every 8 hours" (null if not specified)
- duration: e.g. "5 days", "1 month" (null if not specified)
- route: "oral", "IV", "topical", etc. (null if not specified)
- source: "hospital" if it's an IV drug or clearly given during admission, otherwise "current"

Recognise common Indian abbreviations (e.g. BD = twice daily, TDS = three times daily, HS = at night, SOS = as needed).

Return ONLY a valid JSON array. No prose, no markdown fences. Example:
[{{"name": "Paracetamol", "dosage": "500mg", "frequency": "TDS", "duration": "3 days", "route": "oral", "source": "current"}}]

If no medications found, return an empty array: []
"""

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        # Claude returns content as a list of blocks
        content_blocks = data.get("content", [])
        text = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        ).strip()

        # Strip markdown fences if Claude wrapped JSON in them
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        drugs = json.loads(text)
        if not isinstance(drugs, list):
            logger.warning("Claude returned non-list: %s", type(drugs))
            return []
        return drugs
    except json.JSONDecodeError as e:
        logger.error("Claude returned invalid JSON: %s", e)
        return []
    except Exception as e:
        logger.error("Claude drug extraction failed: %s", e)
        return []


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/")
def root():
    """Health check / welcome."""
    return {
        "service": "MedRecord OCR",
        "status": "running",
        "google_vision_configured": bool(GOOGLE_VISION_KEY),
        "claude_configured": bool(ANTHROPIC_API_KEY),
        "endpoints": ["/health", "/ocr", "/extract-drugs", "/docs"],
    }


@app.get("/health")
def health():
    """Liveness probe for Railway."""
    return {"status": "healthy"}


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    """Plain OCR — returns extracted text from an image or PDF page."""
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        text, engine = extract_text_with_fallback(image_bytes)
        return OCRResponse(success=True, text=text, engine=engine)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("OCR failed")
        return OCRResponse(success=False, text="", engine="none", error=str(e))


@app.post("/extract-drugs", response_model=DrugExtractionResponse)
async def extract_drugs_endpoint(file: UploadFile = File(...)):
    """Full pipeline: OCR -> Claude drug extraction. Used for prescriptions."""
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        text, engine = extract_text_with_fallback(image_bytes)
        drugs = extract_drugs_with_claude(text)

        return DrugExtractionResponse(
            success=True,
            text=text,
            engine=engine,
            drugs=drugs,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Drug extraction failed")
        return DrugExtractionResponse(
            success=False, text="", engine="none", drugs=[], error=str(e)
        )


# ------------------------------------------------------------------
# Local dev entrypoint (Railway uses uvicorn from start command)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)