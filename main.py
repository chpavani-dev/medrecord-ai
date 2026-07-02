"""
MedRecord OCR Microservice
FastAPI + Google Vision (REST) + EasyOCR fallback + Claude Haiku 4.5

vv1.9.0 — Precision fix for handwritten prescriptions:
  - CHANGED: Tightened vision prompt to NOT extract vitals/diagnosis/instructions as drugs
  - NEW: filter_non_medications() drops items that are clearly vitals
         (BP, PR, HR, Wt, Ht, SpO2, Temp, RR) before returning
  - Vision was capturing too much from handwritten Rx — now reads only meds

  v1.8.1 — Always-vision for image prescriptions:
  - CHANGED: Image prescriptions ALWAYS use Claude vision (was cascade with text fallback)
  - PDFs still use text extraction (their text is already clean)
  - Cascade approach returned partial/incorrect data for handwritten Rx;
    always-vision is simpler and more reliable
  - Vision path captures dosage + frequency + drug names holistically

  v1.8.0 — Handwritten prescription auto-detect (cascade):
  - NEW: extract_drugs_with_claude_vision(image_bytes) sends image directly to Claude
  - NEW: /ocr/prescription cascades — tries cheap OCR+Haiku first; if <2 drugs detected,
         falls back to Claude vision (handles handwriting much better)
  - NEW: avg_confidence reflects extraction quality (high/medium/low)
         App can show a "please verify" banner only when confidence is low

  v1.7.0 — PDF support:
  - Native PDF parsing via pdfplumber (digital PDFs)
  - Scanned PDF fallback via pdf2image + OCR (rasterizes pages)
  - All /ocr, /ocr/prescription, /ocr/report endpoints accept PDFs
  - Content-type dispatcher routes uploads to the right extractor

  Carries forward:
  - v1.6.0: prescription_date extraction in /extract-drugs and /ocr/prescription
  - v1.5.0: /ocr/report multi-panel parsing with abnormal_findings
  - v1.4.0: /ocr/prescription, /ocr/lab, /extract_drugs aliases
  - v1.3.0: drug_name, type, avg_confidence app-compat fields
  - v1.2.0: deterministic IV->hospital classification
"""

import os
import base64
import io
import json
import logging
import re
from datetime import datetime
from typing import Optional

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------------------------------------------------------------------
# Logging
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
    logger.warning("ANTHROPIC_API_KEY not set — Claude parsing will be skipped")

# ------------------------------------------------------------------
# Standard 9 trends metrics (Tier 1)
# ------------------------------------------------------------------
STANDARD_METRICS = {
    "hba1c", "glucose", "hemoglobin", "haemoglobin", "tsh",
    "cholesterol", "ldl", "hdl", "triglycerides", "creatinine",
}

# ------------------------------------------------------------------
# Test name normalization
# ------------------------------------------------------------------
TEST_NAME_NORMALIZATION = {
    "creatinine": "Creatinine", "s. creatinine": "Creatinine",
    "serum creatinine": "Creatinine", "creat": "Creatinine",
    "hemoglobin": "Hemoglobin", "haemoglobin": "Hemoglobin",
    "hgb": "Hemoglobin", "hb": "Hemoglobin",
    "hba1c": "HbA1c", "glycated hemoglobin": "HbA1c", "glycosylated hemoglobin": "HbA1c",
    "glucose": "Glucose", "fasting glucose": "Fasting Glucose", "fbs": "Fasting Glucose",
    "ppbs": "Post Prandial Glucose", "post prandial glucose": "Post Prandial Glucose",
    "rbs": "Random Glucose",
    "tsh": "TSH", "thyroid stimulating hormone": "TSH",
    "t3": "T3", "t4": "T4", "free t3": "Free T3", "free t4": "Free T4",
    "cholesterol": "Total Cholesterol", "total cholesterol": "Total Cholesterol",
    "ldl": "LDL Cholesterol", "ldl cholesterol": "LDL Cholesterol",
    "hdl": "HDL Cholesterol", "hdl cholesterol": "HDL Cholesterol",
    "triglycerides": "Triglycerides", "tg": "Triglycerides",
    "uric acid": "Uric Acid",
    "vitamin d": "Vitamin D", "25 oh vitamin d": "Vitamin D", "25-oh vitamin d": "Vitamin D",
    "vitamin b12": "Vitamin B12", "b12": "Vitamin B12",
    "wbc": "WBC", "white blood cells": "WBC",
    "rbc": "RBC", "red blood cells": "RBC",
    "platelets": "Platelets", "plt": "Platelets",
    "esr": "ESR", "crp": "CRP",
    "sgot": "SGOT (AST)", "ast": "SGOT (AST)",
    "sgpt": "SGPT (ALT)", "alt": "SGPT (ALT)",
    "bilirubin": "Bilirubin Total", "total bilirubin": "Bilirubin Total",
    "urea": "Urea", "blood urea": "Urea", "bun": "BUN",
    "sodium": "Sodium", "na": "Sodium",
    "potassium": "Potassium", "k": "Potassium",
}


def normalize_test_name(raw_name: str) -> str:
    if not raw_name:
        return raw_name
    key = raw_name.strip().lower()
    key = re.sub(r"^[sp]\.\s*", "", key)
    if key in TEST_NAME_NORMALIZATION:
        return TEST_NAME_NORMALIZATION[key]
    return raw_name.strip().title()


# ------------------------------------------------------------------
# Date normalization
# ------------------------------------------------------------------
def normalize_date(raw_date) -> Optional[str]:
    if not raw_date or not isinstance(raw_date, str):
        return None
    s = raw_date.strip()
    if not s or s.lower() in ("null", "none", "n/a"):
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    patterns = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%d/%m/%y", "%d-%m-%y",
        "%d %B %Y", "%d %b %Y",
        "%B %d, %Y", "%b %d, %Y",
        "%Y/%m/%d", "%Y-%m-%d",
        "%m/%d/%Y", "%m-%d-%Y",
    ]
    for p in patterns:
        try:
            dt = datetime.strptime(s, p)
            now = datetime.now()
            if dt.year < 1950 or dt > datetime(now.year + 1, 12, 31):
                continue
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    logger.warning("Could not normalize date: %s", raw_date)
    return None
# ================================================================
# WATERMARK REMOVAL FUNCTIONS - add to main.py after the imports
# ================================================================

def strip_text_watermarks(text: str) -> str:
    """
    Remove repeated watermark words from pdfplumber-extracted text.
    Watermarks in digital PDFs appear as the same word/phrase repeated
    many times scattered throughout the text.
    Strategy: find words that appear suspiciously often relative to
    total word count, and remove them if they look like watermarks.
    """
    if not text:
        return text

    import re
    from collections import Counter

    # Split into words, preserve structure
    words = text.split()
    if len(words) < 20:
        return text  # Too short to analyze

    # Count word frequencies (case-insensitive)
    word_counts = Counter(w.lower().strip('.,;:()[]') for w in words if len(w) > 2)
    total_words = len(words)

    # A word is a watermark candidate if it appears > 3% of total words
    # AND appears more than 5 times (avoid false positives on short docs)
    watermark_words = set()
    for word, count in word_counts.items():
        frequency = count / total_words
        if frequency > 0.03 and count > 5:
            # Additional check: watermark words are usually all-caps or title case
            # and are NOT common medical/lab words
            COMMON_LAB_WORDS = {
                'the', 'and', 'for', 'with', 'not', 'are', 'was', 'result',
                'normal', 'range', 'value', 'test', 'report', 'blood', 'urine',
                'serum', 'level', 'high', 'low', 'date', 'name', 'unit', 'ref'
            }
            if word not in COMMON_LAB_WORDS:
                watermark_words.add(word)
                logger.info("Detected watermark word: '%s' (appears %d times, %.1f%%)",
                           word, count, frequency * 100)

    if not watermark_words:
        return text

    # Remove watermark words from text (case-insensitive, whole word only)
    cleaned = text
    for wm_word in watermark_words:
        pattern = re.compile(r'\b' + re.escape(wm_word) + r'\b', re.IGNORECASE)
        cleaned = pattern.sub('', cleaned)

    # Clean up extra whitespace left behind
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    logger.info("Watermark removal: %d suspect words stripped from PDF text", len(watermark_words))
    return cleaned.strip()


def preprocess_image_remove_watermark(image_bytes: bytes) -> bytes:
    """
    Remove diagonal/repeated watermarks from scanned lab report images
    before OCR. Uses OpenCV to detect and mask diagonal text patterns.
    Returns cleaned image bytes (JPEG).
    """
    try:
        import numpy as np
        import cv2

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Could not decode image for watermark removal")
            return image_bytes

        # Convert to grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 1: Detect text-like regions using adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # Step 2: Find connected components (text blobs)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh, connectivity=8
        )

        # Step 3: Identify diagonal/watermark components
        # Watermark text tends to be:
        # - Medium sized (not too small like noise, not too large like headers)
        # - Arranged diagonally across the page
        # - Lower contrast than actual report text (lighter gray)
        h, w = img.shape[:2]
        page_area = h * w
        watermark_mask = np.zeros(gray.shape, dtype=np.uint8)

        # Detect light gray regions (watermark is usually 30-70% gray)
        # while report text is near black (0-20% gray)
        lower_gray = np.array([150])  # Light gray lower bound
        upper_gray = np.array([220])  # Light gray upper bound
        gray_mask = cv2.inRange(gray, lower_gray[0], upper_gray[0])

        # Dilate to connect nearby watermark components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray_mask_dilated = cv2.dilate(gray_mask, kernel, iterations=2)

        # Find contours in the gray mask
        contours, _ = cv2.findContours(
            gray_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        watermark_regions_found = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by area - watermark regions are medium-sized
            if area < 500 or area > page_area * 0.15:
                continue

            # Check if region spans a significant diagonal
            rect = cv2.boundingRect(contour)
            rx, ry, rw, rh = rect
            aspect_ratio = rw / max(rh, 1)

            # Watermarks often have unusual aspect ratios (very wide or diagonal)
            if aspect_ratio > 3 or (rw > w * 0.2 and rh > h * 0.05):
                # Fill this region with white in the original image
                cv2.fillPoly(img, [contour], (255, 255, 255))
                watermark_regions_found += 1

        if watermark_regions_found > 0:
            logger.info("Image watermark removal: masked %d regions", watermark_regions_found)
        else:
            logger.info("Image watermark removal: no clear watermark regions detected")
            # Return original if nothing found - don't degrade image quality unnecessarily
            return image_bytes

        # Encode back to JPEG
        success, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            logger.warning("Could not re-encode image after watermark removal")
            return image_bytes

        return encoded.tobytes()

    except Exception as e:
        logger.warning("Image watermark removal failed: %s — using original image", e)
        return image_bytes  # Always fall back to original

# ------------------------------------------------------------------
# EasyOCR (lazy-loaded fallback)
# ------------------------------------------------------------------
_easyocr_reader = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        logger.info("Loading OCR engine...")
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
        logger.info("OCR engine ready!")
    return _easyocr_reader


# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(
    title="MedRecord OCR Service",
    description="OCR + AI parsing for prescriptions and lab reports (images + PDFs)",
    version="1.9.2",
)

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
    engine: str
    error: Optional[str] = None

class DrugExtractionResponse(BaseModel):
    success: bool
    text: str
    engine: str
    drugs: list
    prescription_date: Optional[str] = None
    doctor_name: Optional[str] = None
    hospital_name: Optional[str] = None
    avg_confidence: int = 0
    error: Optional[str] = None

class LabReportResponse(BaseModel):
    success: bool
    text: str
    engine: str
    lab_name: Optional[str] = None
    report_date: Optional[str] = None
    patient_name: Optional[str] = None
    panels: list = []
    abnormal_findings: list = []
    avg_confidence: int = 0
    error: Optional[str] = None


# ------------------------------------------------------------------
# OCR — Google Vision via REST
# ------------------------------------------------------------------
def extract_text_google_vision(image_bytes: bytes) -> str:
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
    responses = result.get("responses", [])
    if not responses:
        return ""
    first = responses[0]
    if "error" in first:
        err = first["error"]
        raise RuntimeError(f"Google Vision API error: {err.get('message', err)}")
    text_annotations = first.get("textAnnotations", [])
    if not text_annotations:
        return ""
    return text_annotations[0].get("description", "")


def extract_text_easyocr(image_bytes: bytes) -> str:
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
    """OCR an image (jpeg/png) with Google Vision -> EasyOCR fallback."""
    try:
        text = extract_text_google_vision(image_bytes)
        if text and text.strip():
            logger.info("Google Vision succeeded (%d chars)", len(text))
            return text, "google_vision"
    except Exception as e:
        logger.warning("Google Vision failed: %s — falling back to EasyOCR", e)

    try:
        text = extract_text_easyocr(image_bytes)
        logger.info("EasyOCR succeeded (%d chars)", len(text))
        return text, "easyocr"
    except Exception as e:
        logger.error("EasyOCR also failed: %s", e)
        raise


# ==================================================================
# NEW v1.7.0 — PDF text extraction
# ==================================================================
def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, str]:
    """
    Extract text from a PDF. Tries pdfplumber first (works for digital PDFs).
    Falls back to rasterizing pages and OCR-ing them (for scanned PDFs).
    Returns (text, engine) where engine is one of:
      - "pdf_text"      — pdfplumber direct text
      - "pdf_ocr"       — rasterized + Google Vision / EasyOCR
      - "pdf_failed"    — couldn't extract anything
    """
    # ---- Strategy 1: pdfplumber (digital PDFs) ----
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text_parts = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            full_text = "\n".join(text_parts).strip()
            if full_text and len(full_text) > 50:
                logger.info("pdfplumber extracted %d chars from %d pages",
                            len(full_text), len(pdf.pages))
                full_text = strip_text_watermarks(full_text)
                return full_text, "pdf_text"
            else:
                logger.info("pdfplumber got only %d chars — likely a scanned PDF, falling back to OCR",
                            len(full_text))
    except Exception as e:
        logger.warning("pdfplumber failed: %s — falling back to OCR", e)
    # ---- Strategy 2: rasterize each page and OCR ----
    try:
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(pdf_bytes, dpi=200)
        # Cap at 10 pages to avoid runaway costs
        if len(images) > 10:
            logger.warning("PDF has %d pages — capping at first 10", len(images))
            images = images[:10]

        all_text = []
        last_engine = "easyocr"
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            page_bytes = buf.getvalue()
            try:
                page_text, engine = extract_text_with_fallback(page_bytes)
                last_engine = engine
                if page_text:
                    all_text.append(page_text)
            except Exception as page_e:
                logger.warning("Page %d OCR failed: %s", i + 1, page_e)
                continue

        full_text = "\n".join(all_text).strip()
        if full_text:
            logger.info("PDF OCR extracted %d chars from %d page(s) using %s",
                        len(full_text), len(images), last_engine)
            return full_text, "pdf_ocr"
    except Exception as e:
        logger.error("PDF rasterization failed: %s", e)

    logger.error("All PDF extraction strategies failed")
    return "", "pdf_failed"


def detect_pdf(file: UploadFile, file_bytes: bytes) -> bool:
    """Detect if uploaded file is a PDF based on content-type or magic bytes."""
    content_type = (file.content_type or "").lower()
    if "pdf" in content_type:
        return True
    filename = (file.filename or "").lower()
    if filename.endswith(".pdf"):
        return True
    # Magic bytes: PDFs start with "%PDF-"
    if file_bytes[:5] == b"%PDF-":
        return True
    return False


def get_text_from_upload(file: UploadFile, file_bytes: bytes) -> tuple[str, str]:
    """
    Unified text extraction dispatcher.
    Returns (text, engine) for any supported upload type.
    """
    if detect_pdf(file, file_bytes):
        return extract_text_from_pdf(file_bytes)
    return extract_text_with_fallback(file_bytes)


# ==================================================================
# DRUG EXTRACTION
# ==================================================================
DRUG_EXTRACTION_PROMPT = """You are a clinical pharmacist parsing an Indian prescription or hospital discharge summary.

OCR TEXT:
---
{ocr_text}
---

# YOUR TASK

Return a JSON object with TWO parts:

## Part 1: METADATA — find these fields
- `prescription_date`: The date the prescription was written / discharge date / report date.
  - Look for "Date:", "Issued on:", "Discharge Date:", "Admission Date:", or any date near the doctor's signature
  - For discharge summaries, prefer the DISCHARGE date (not admission date)
  - Indian format is often DD/MM/YYYY or DD-MM-YYYY (e.g., "28/04/2026" = April 28, 2026)
  - Normalize to YYYY-MM-DD format (e.g., "2026-04-28")
  - If multiple dates exist, prefer the one closest to "Discharge", "Date", or doctor signature
  - If no date found, use null
- `doctor_name`: Name of the prescribing doctor (e.g., "Dr. Rajesh Kumar"). Null if not present.
- `hospital_name`: Name of the hospital or clinic. Null if not present.

## Part 2: DRUGS — extract every medication

Indian discharge summaries have TWO sections:

### HOSPITAL MEDICATIONS (during admission)
- IV / Intravenous infusions, Injections (Inj.)
- Chemo drugs, premedications
- Set "source": "hospital"

### DISCHARGE / TAKE-HOME MEDICATIONS
- TAB./CAP./SYP. prefixes, oral pills
- Indian dosing: 1-0-0, 1-1-1, BD, TDS, HS, SOS
- Set "source": "current"

# OUTPUT FORMAT

Return ONLY a JSON object. No prose, no markdown fences:
{{
  "prescription_date": "YYYY-MM-DD or null",
  "doctor_name": "Dr. Name or null",
  "hospital_name": "Hospital or null",
  "drugs": [
    {{
      "name": "drug name",
      "dosage": "e.g. '500mg' or null",
      "frequency": "e.g. 'BD', '1-0-0' or null",
      "duration": "e.g. '5 days' or null",
      "route": "oral | IV | subcutaneous | topical | inhalation",
      "source": "hospital | current"
    }}
  ]
}}

# ABBREVIATIONS
BD=twice daily, TDS=three times daily, QID=four times, HS=bedtime, SOS=as needed
1-0-0=morning, 1-1-1=morning/afternoon/night, 1-0-1=morning/night

Now parse the document and return the JSON object:"""


def extract_drugs_with_claude(ocr_text: str) -> dict:
    if not ANTHROPIC_API_KEY or not ocr_text or not ocr_text.strip():
        return {"drugs": [], "prescription_date": None, "doctor_name": None, "hospital_name": None}

    prompt = DRUG_EXTRACTION_PROMPT.format(ocr_text=ocr_text)
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }

    text = ""
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=payload, timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        content_blocks = data.get("content", [])
        text = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        ).strip()

        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        result = json.loads(text)
        if not isinstance(result, dict):
            return {"drugs": [], "prescription_date": None, "doctor_name": None, "hospital_name": None}

        drugs = result.get("drugs", []) if isinstance(result.get("drugs"), list) else []
        drugs = post_process_drugs(drugs)
        drugs = filter_non_medications(drugs)
        drugs = add_drug_app_compat_fields(drugs)

        rx_date = normalize_date(result.get("prescription_date"))

        logger.info("Drugs: %d total (%d hospital, %d current) | date: %s | hospital: %s",
                    len(drugs),
                    sum(1 for d in drugs if d.get("source") == "hospital"),
                    sum(1 for d in drugs if d.get("source") == "current"),
                    rx_date, result.get("hospital_name"))

        return {
            "drugs": drugs,
            "prescription_date": rx_date,
            "doctor_name": result.get("doctor_name"),
            "hospital_name": result.get("hospital_name"),
        }
    except json.JSONDecodeError as e:
        logger.error("Claude returned invalid JSON for drugs: %s | text: %s", e, text[:300])
        return {"drugs": [], "prescription_date": None, "doctor_name": None, "hospital_name": None}
    except Exception as e:
        logger.error("Claude drug extraction failed: %s", e)
        return {"drugs": [], "prescription_date": None, "doctor_name": None, "hospital_name": None}


# ==================================================================
# NEW v1.8.0 — Claude vision direct path for handwritten prescriptions
# ==================================================================
DRUG_VISION_PROMPT = """You are a clinical pharmacist looking at a photograph of an Indian doctor's prescription or hospital discharge summary. The handwriting may be messy. Read it carefully and extract ONLY medications.

# YOUR TASK

Return a JSON object with two parts:

## Part 1: METADATA
- prescription_date: Visit/consultation/discharge date in YYYY-MM-DD format. Look in printed header AND handwritten notes.
- doctor_name: Prescribing doctor's name (e.g., "Dr. Rajesh Kumar"). Null if not present.
- hospital_name: Hospital or clinic name. Null if not present.

## Part 2: MEDICATIONS ONLY — extract real prescribed drugs

A medication is something the patient TAKES. Usually:
- Prefixed by "Tab.", "Cap.", "Syp.", "Inj.", "Sol.", "Drops"
- Common Indian brand names: Deplatt, Ecospirin, Telma, Atorva, Metoprolol, Pan, Crocin, Dolo, Glycomet, Amlong, Concor, Eptus, Dytor, Hyponat, L-Montus, Valentas
- Has a dosage in mg/mcg/ml AND/OR a frequency

# CRITICAL — DO NOT INCLUDE THESE AS MEDICATIONS:

❌ VITALS & MEASUREMENTS (NEVER drugs):
   - BP, B.P, Blood Pressure (e.g., "BP 130/85 mmHg")
   - PR, P.R, Pulse Rate, HR, Heart Rate (e.g., "PR 91/min")
   - SpO2, Oxygen Saturation (e.g., "SpO2 98%")
   - Wt, Weight (e.g., "Wt 78 kg")
   - Ht, Height
   - Temp, Temperature
   - RR, Respiratory Rate
   - BMI

❌ DIAGNOSIS & EXAM FINDINGS (NEVER drugs):
   - CABG, CVS, RS, CNS, P/A (anatomy/exam abbreviations)
   - Hypertension, Diabetes, "Mod LV Dysfunction"
   - Any line containing "Dysfunction", "Syndrome", "Disease"
   - "S1 S2 normal", "No murmurs"

❌ INSTRUCTIONS & LIFESTYLE (NEVER drugs):
   - "Fluid restriction <1L/day"
   - "Salt restriction"
   - "Walk 30 min daily"
   - "Review after X days"
   - "Follow up"

❌ LAB TESTS (these are ORDERS not drugs):
   - CBC, RFT, LFT, KFT, ECG, ECHO, X-Ray, MRI, CT, Ultrasound
   - Even if they appear in a numbered list

# INDIAN DOSING NOTATION (for frequency field)
- 1-0-0 (morning only)
- 1-0-1 (morning + night)
- 1-1-1 (morning + afternoon + night)
- BD = twice daily
- TDS / TID = three times daily
- QID = four times daily
- HS = at bedtime
- SOS = as needed
- OD = once daily
- PO OD = orally once daily

## CONFIDENCE
For each drug, include "confidence": "high", "medium", or "low":
- "high" = name AND dose AND frequency clearly legible
- "medium" = at least one field required interpretation
- "low" = significant guesswork on name or dose

# OUTPUT FORMAT

Return ONLY JSON, no prose, no markdown fences:
{
  "prescription_date": "YYYY-MM-DD or null",
  "doctor_name": "string or null",
  "hospital_name": "string or null",
  "drugs": [
    {
      "name": "drug name as written (e.g., 'Tab. Deplatt-CV')",
      "dosage": "e.g. '500mg' or null",
      "frequency": "BD | TDS | 1-0-1 | 1-1-1 | HS | SOS | OD | etc or null",
      "duration": "e.g. '5 days' or null",
      "route": "oral | IV | subcutaneous | topical | inhalation | null",
      "source": "current | hospital",
      "confidence": "high | medium | low"
    }
  ]
}

# DOUBLE-CHECK BEFORE RETURNING

Before you return, look at every item in your drugs list and ask:
- Is this something a pharmacist would dispense? (If no → REMOVE)
- Does the name contain "BP", "PR", "HR", "Wt", "Ht", "SpO2", "Temp", "BMI"? (If yes → REMOVE)
- Is this a vital sign measurement like "130/85" or "91/min" or "98%"? (If yes → REMOVE)

Read the image carefully and return the JSON object:"""


def extract_drugs_with_claude_vision(image_bytes: bytes) -> dict:
    """
    Send image DIRECTLY to Claude Haiku vision — no OCR step.
    Designed for handwritten prescriptions where OCR fails.
    Returns same shape as extract_drugs_with_claude().
    """
    if not ANTHROPIC_API_KEY or not image_bytes:
        return {"drugs": [], "prescription_date": None, "doctor_name": None,
                "hospital_name": None, "method": "vision_failed"}

    # Encode image as base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 4096,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": DRUG_VISION_PROMPT},
            ],
        }],
    }

    text = ""
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=payload, timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        content_blocks = data.get("content", [])
        text = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        ).strip()

        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        result = json.loads(text)
        if not isinstance(result, dict):
            return {"drugs": [], "prescription_date": None, "doctor_name": None,
                    "hospital_name": None, "method": "vision_failed"}

        drugs = result.get("drugs", []) if isinstance(result.get("drugs"), list) else []
        drugs = post_process_drugs(drugs)
        drugs = filter_non_medications(drugs)
        drugs = add_drug_app_compat_fields(drugs)

        rx_date = normalize_date(result.get("prescription_date"))

        logger.info("Vision extraction: %d drugs | high:%d medium:%d low:%d | date:%s",
                    len(drugs),
                    sum(1 for d in drugs if d.get("confidence") == "high"),
                    sum(1 for d in drugs if d.get("confidence") == "medium"),
                    sum(1 for d in drugs if d.get("confidence") == "low"),
                    rx_date)

        return {
            "drugs": drugs,
            "prescription_date": rx_date,
            "doctor_name": result.get("doctor_name"),
            "hospital_name": result.get("hospital_name"),
            "method": "vision",
        }
    except json.JSONDecodeError as e:
        logger.error("Claude vision returned invalid JSON: %s | text: %s", e, text[:300])
        return {"drugs": [], "prescription_date": None, "doctor_name": None,
                "hospital_name": None, "method": "vision_failed"}
    except Exception as e:
        logger.error("Claude vision extraction failed: %s", e)
        return {"drugs": [], "prescription_date": None, "doctor_name": None,
                "hospital_name": None, "method": "vision_failed"}


def compute_avg_confidence(drugs: list, base_engine_confidence: int) -> int:
    """
    Combine engine quality with per-drug confidence to produce a 0-100 score.
    App uses this to decide whether to show a 'please verify' banner.
    """
    if not drugs:
        return 0
    levels = {"high": 95, "medium": 80, "low": 60}
    drug_scores = [levels.get(d.get("confidence", "high"), 90) for d in drugs]
    drug_avg = sum(drug_scores) / len(drug_scores)
    # Blend with engine confidence (60/40 weighted toward drug-level)
    return int(0.6 * drug_avg + 0.4 * base_engine_confidence)


def filter_non_medications(drugs: list) -> list:
    """
    Safety-net filter. Drops items that are clearly vitals, diagnosis tokens,
    or lab test orders rather than medications. Used after Claude vision
    extraction to catch anything the prompt missed.
    """
    # Tokens that, if they appear as the WHOLE NAME (case-insensitive),
    # mean this isn't a drug. Order-sensitive: check exact name first.
    NON_DRUG_NAMES = {
        # Vitals
        "bp", "b.p", "b.p.", "blood pressure",
        "pr", "p.r", "p.r.", "pulse", "pulse rate",
        "hr", "h.r", "heart rate",
        "spo2", "sp02", "spo 2", "oxygen saturation",
        "wt", "weight",
        "ht", "height",
        "temp", "temperature",
        "rr", "r.r", "respiratory rate",
        "bmi",
        # Exam findings
        "cvs", "rs", "cns", "p/a", "cabg",
        # Lab tests (these are orders, not drugs)
        "cbc", "rft", "lft", "kft", "ecg", "echo", "x-ray", "mri", "ct", "ultrasound", "usg",
    }

    # Phrase fragments that, if present anywhere in name/dosage, mean it's not a drug
    NON_DRUG_PHRASES = [
        "mmhg", "/min", "kg", "%spo", "% spo",
        "dysfunction", "syndrome",
        "fluid restriction", "salt restriction",
        "review after", "follow up", "f/u",
    ]

    cleaned = []
    for d in drugs:
        if not isinstance(d, dict):
            continue
        raw_name = str(d.get("name", "") or "").strip()
        if not raw_name:
            continue

        name_lower = raw_name.lower()
        # Strip leading "tab.", "cap.", "syp." etc to compare the core token
        core_name = re.sub(r"^\s*(tab|cap|syp|syr|inj|sol|drops)\.?\s+", "", name_lower).strip()

        # Reject if the entire name is a known non-drug token
        if core_name in NON_DRUG_NAMES:
            logger.info("Filtered non-medication: %s (exact match)", raw_name)
            continue

        # Reject if any non-drug phrase is in the name or dosage field
        combined = (raw_name + " " + str(d.get("dosage", "") or "")).lower()
        if any(phrase in combined for phrase in NON_DRUG_PHRASES):
            logger.info("Filtered non-medication: %s (phrase match)", raw_name)
            continue

        # Reject obvious vital-sign value patterns in the name
        # e.g. "130/85", "98%", "91/min"
        if re.search(r"\b\d{2,3}/\d{2,3}\b|\b\d{2,3}\s*%\b|\b\d{2,3}\s*/\s*min\b", raw_name):
            logger.info("Filtered non-medication: %s (vital-value pattern)", raw_name)
            continue

        cleaned.append(d)

    return cleaned


def post_process_drugs(drugs: list) -> list:
    cleaned = []
    for d in drugs:
        if not isinstance(d, dict) or not d.get("name"):
            continue
        name = str(d.get("name", "")).strip()
        route = str(d.get("route", "") or "").lower()
        is_iv = any(kw in route for kw in ["iv", "intravenous", "infusion", "subcutaneous", "inj"])
        is_inj = bool(re.match(r"^\s*inj\.?\b", name, re.IGNORECASE))
        if is_iv or is_inj:
            d["source"] = "hospital"
        is_oral = bool(re.match(r"^\s*(tab|cap|syp|syr)\.?\b", name, re.IGNORECASE))
        if is_oral:
            d["source"] = "current"
            if not d.get("route"):
                d["route"] = "oral"
        if not d.get("source"):
            d["source"] = "current"
        cleaned.append(d)
    return cleaned


def add_drug_app_compat_fields(drugs: list) -> list:
    for d in drugs:
        if "name" in d and "drug_name" not in d:
            d["drug_name"] = d["name"]
        source = d.get("source", "current")
        d["type"] = "hospital" if source == "hospital" else "outpatient"
        if not d.get("category"):
            d["category"] = "Medication"
    return drugs


# ==================================================================
# LAB REPORT EXTRACTION
# ==================================================================
LAB_REPORT_PROMPT = """You are a clinical lab technician parsing a lab report from an Indian diagnostic lab (SRL, Thyrocare, Metropolis, Dr. Lal Path Labs, Apollo, Vijaya, etc.).

OCR TEXT:
---
{ocr_text}
---

# YOUR TASK

Parse this lab report into STRUCTURED JSON. A SINGLE upload may contain MULTIPLE PANELS — split them into separate panel objects.

# DATE EXTRACTION (CRITICAL)

Find `report_date`:
- Look for "Reported on:", "Report Date:", "Date:", "Sample collected:", "Tested on:"
- If both COLLECTION and REPORT date exist, prefer REPORT date
- Indian format is often DD/MM/YYYY (e.g., "28/04/2026" = April 28, 2026)
- Normalize to YYYY-MM-DD format
- If no date found, use null

# CATEGORIES

Auto-categorize each panel into ONE of:
- "Blood" — CBC, LFT, RFT/KFT, Lipid Panel, HbA1c, Thyroid, Glucose, Vitamin D/B12, Iron, Electrolytes
- "Urine" — Urinalysis, Microalbumin, 24-hr protein, Urine culture
- "Imaging" — X-Ray, CT, MRI, Ultrasound, Mammogram, Echo
- "Pathology" — Biopsy, FNAC, Cytology, Histopathology
- "Cardiac" — ECG, 2D Echo, TMT, Holter
- "Other"

# COMMON PANELS

- "Complete Blood Count" / "CBC" / "Hemogram" -> Blood
- "Lipid Profile" / "Lipid Panel" -> Blood
- "Liver Function Test" / "LFT" -> Blood
- "Renal Function Test" / "RFT" / "KFT" -> Blood
- "Thyroid Profile" -> Blood
- "HbA1c" -> Blood
- "Urinalysis" / "Urine Routine" -> Urine

If panel name unclear, infer from tests present (Hb + WBC + Platelets = "Complete Blood Count").

# OUTPUT FORMAT

Return ONLY a JSON object. No prose, no markdown fences:
{{
  "lab_name": "name or null",
  "report_date": "YYYY-MM-DD or null",
  "patient_name": "name or null",
  "panels": [
    {{
      "panel_name": "Complete Blood Count",
      "category": "Blood",
      "tests": [
        {{
          "name": "Hemoglobin",
          "value": 12.5,
          "unit": "g/dL",
          "normal_range": "13.0-17.0",
          "flag": "low"
        }}
      ]
    }}
  ]
}}

# RULES

1. **value**: Number for numeric (12.5, 7800). String for qualitative ("Yellow", "Trace"). Null if missing.
2. **unit**: Exactly as printed. Null if no unit.
3. **normal_range**: Extract from report. Null if not present.
4. **flag**: "low" | "high" | "critical_low" | "critical_high" | "abnormal" | "normal" | null
5. **name**: Use printed name. Don't normalize — backend handles that.
6. Extract EVERY test you see. Don't skip.

Now parse the report and return the JSON object:"""


STANDARD_REFERENCE_RANGES = {
    "Hemoglobin":        {"low": 12.0, "high": 17.0, "unit": "g/dL", "critical_low": 7.0,  "critical_high": 20.0},
    "HbA1c":             {"low": 4.0,  "high": 5.7,  "unit": "%",    "critical_low": None, "critical_high": 12.0},
    "Glucose":           {"low": 70,   "high": 100,  "unit": "mg/dL","critical_low": 50,   "critical_high": 400},
    "Fasting Glucose":   {"low": 70,   "high": 100,  "unit": "mg/dL","critical_low": 50,   "critical_high": 400},
    "TSH":               {"low": 0.4,  "high": 4.5,  "unit": "uIU/mL","critical_low": None,"critical_high": 50},
    "Total Cholesterol": {"low": None, "high": 200,  "unit": "mg/dL","critical_low": None, "critical_high": 400},
    "LDL Cholesterol":   {"low": None, "high": 100,  "unit": "mg/dL","critical_low": None, "critical_high": None},
    "HDL Cholesterol":   {"low": 40,   "high": None, "unit": "mg/dL","critical_low": None, "critical_high": None},
    "Triglycerides":     {"low": None, "high": 150,  "unit": "mg/dL","critical_low": None, "critical_high": 1000},
    "Creatinine":        {"low": 0.6,  "high": 1.3,  "unit": "mg/dL","critical_low": None, "critical_high": 5.0},
    "Uric Acid":         {"low": 3.5,  "high": 7.2,  "unit": "mg/dL","critical_low": None, "critical_high": None},
    "Vitamin D":         {"low": 30,   "high": 100,  "unit": "ng/mL","critical_low": None, "critical_high": None},
    "Vitamin B12":       {"low": 200,  "high": 900,  "unit": "pg/mL","critical_low": None, "critical_high": None},
    "Potassium":         {"low": 3.5,  "high": 5.0,  "unit": "mEq/L","critical_low": 2.5,  "critical_high": 6.0},
    "Sodium":            {"low": 135,  "high": 145,  "unit": "mEq/L","critical_low": 120,  "critical_high": 160},
}


def determine_flag(test_name: str, value, normal_range: str) -> Optional[str]:
    if value is None or not isinstance(value, (int, float)):
        return None

    canonical = normalize_test_name(test_name)
    if canonical in STANDARD_REFERENCE_RANGES:
        ref = STANDARD_REFERENCE_RANGES[canonical]
        if ref["critical_low"] is not None and value <= ref["critical_low"]:
            return "critical_low"
        if ref["critical_high"] is not None and value >= ref["critical_high"]:
            return "critical_high"
        if ref["low"] is not None and value < ref["low"]:
            return "low"
        if ref["high"] is not None and value > ref["high"]:
            return "high"
        return "normal"

    if normal_range:
        rng = normal_range.strip()
        m = re.match(r"^([\d.]+)\s*[-–]\s*([\d.]+)", rng)
        if m:
            low, high = float(m.group(1)), float(m.group(2))
            if value < low: return "low"
            if value > high: return "high"
            return "normal"
        m = re.match(r"^<\s*([\d.]+)", rng)
        if m:
            return "high" if value > float(m.group(1)) else "normal"
        m = re.match(r"^>\s*([\d.]+)", rng)
        if m:
            return "low" if value < float(m.group(1)) else "normal"
    return None


def parse_lab_report_with_claude(ocr_text: str) -> dict:
    if not ANTHROPIC_API_KEY or not ocr_text or not ocr_text.strip():
        return {"panels": [], "abnormal_findings": []}

    prompt = LAB_REPORT_PROMPT.format(ocr_text=ocr_text)
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 6144,
        "messages": [{"role": "user", "content": prompt}],
    }

    text = ""
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=payload, timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        content_blocks = data.get("content", [])
        text = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        ).strip()

        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        result = json.loads(text)
        if not isinstance(result, dict):
            return {"panels": [], "abnormal_findings": []}

        result["report_date"] = normalize_date(result.get("report_date"))

        result = post_process_lab_report(result)
        return result
    except json.JSONDecodeError as e:
        logger.error("Claude returned invalid JSON for lab: %s", e)
        return {"panels": [], "abnormal_findings": []}
    except Exception as e:
        logger.error("Claude lab parsing failed: %s", e)
        return {"panels": [], "abnormal_findings": []}


def post_process_lab_report(result: dict) -> dict:
    panels = result.get("panels", [])
    if not isinstance(panels, list):
        panels = []

    abnormal_findings = []

    for panel in panels:
        if not isinstance(panel, dict):
            continue
        tests = panel.get("tests", [])
        if not isinstance(tests, list):
            continue

        for t in tests:
            if not isinstance(t, dict):
                continue
            raw_name = t.get("name", "")
            canonical = normalize_test_name(raw_name)
            t["name"] = canonical
            t["original_name"] = raw_name
            t["is_standard_metric"] = canonical.lower() in STANDARD_METRICS

            value = t.get("value")
            normal_range = t.get("normal_range") or ""
            our_flag = determine_flag(canonical, value, normal_range)
            if our_flag is not None:
                t["flag"] = our_flag

            flag = t.get("flag")
            if flag in ("low", "high", "critical_low", "critical_high", "abnormal"):
                abnormal_findings.append({
                    "name": canonical,
                    "value": value,
                    "unit": t.get("unit"),
                    "normal_range": t.get("normal_range"),
                    "flag": flag,
                    "panel": panel.get("panel_name"),
                    "category": panel.get("category"),
                })

    result["panels"] = panels
    result["abnormal_findings"] = abnormal_findings

    n_panels = len(panels)
    n_tests = sum(len(p.get("tests", [])) for p in panels)
    n_abnormal = len(abnormal_findings)
    logger.info("Lab parsed: %d panels, %d tests, %d abnormal | date: %s",
                n_panels, n_tests, n_abnormal, result.get("report_date"))

    return result


# ------------------------------------------------------------------
# Engine -> confidence mapping
# ------------------------------------------------------------------
def confidence_for_engine(engine: str) -> int:
    if engine == "google_vision": return 95
    if engine == "pdf_text":      return 95
    if engine == "pdf_ocr":       return 80
    if engine == "easyocr":       return 75
    return 0


# ------------------------------------------------------------------
# Shared business logic — now dispatches by file type
# ------------------------------------------------------------------
async def _do_ocr(file: UploadFile) -> OCRResponse:
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")
        text, engine = get_text_from_upload(file, file_bytes)
        if engine == "pdf_failed":
            return OCRResponse(success=False, text="", engine=engine, error="Could not extract any text from PDF")
        return OCRResponse(success=True, text=text, engine=engine)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("OCR failed")
        return OCRResponse(success=False, text="", engine="none", error=str(e))


async def _do_extract_drugs(file: UploadFile) -> DrugExtractionResponse:
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        is_pdf = detect_pdf(file, file_bytes)

        # =================================================================
        # PDF PATH: text extraction (PDF text is clean — no need for vision)
        # =================================================================
        if is_pdf:
            text, engine = get_text_from_upload(file, file_bytes)
            if engine == "pdf_failed" or not text:
                return DrugExtractionResponse(
                    success=False, text="", engine=engine,
                    drugs=[], avg_confidence=0,
                    error="Could not extract text from PDF",
                )
            text_result = extract_drugs_with_claude(text)
            text_drugs = text_result.get("drugs", []) or []
            avg_conf = compute_avg_confidence(text_drugs, confidence_for_engine(engine))
            return DrugExtractionResponse(
                success=True, text=text, engine=engine,
                drugs=text_drugs,
                prescription_date=text_result.get("prescription_date"),
                doctor_name=text_result.get("doctor_name"),
                hospital_name=text_result.get("hospital_name"),
                avg_confidence=avg_conf,
            )

        # =================================================================
        # IMAGE PATH: ALWAYS use Claude vision (handles both printed + handwritten)
        # The cheap OCR+Haiku-text path produces unreliable partial results on
        # handwriting (drug names captured but no dosages/frequencies). Vision
        # is roughly cost-neutral and produces holistic, usable results.
        # =================================================================
        logger.info("Image prescription — using Claude vision (always-on)")
        vision_result = extract_drugs_with_claude_vision(file_bytes)
        vision_drugs = vision_result.get("drugs", []) or []

        if not vision_drugs:
            # Vision failed completely — last-ditch fallback to OCR pipeline
            logger.warning("Claude vision returned 0 drugs — falling back to OCR")
            text, engine = get_text_from_upload(file, file_bytes)
            if text:
                text_result = extract_drugs_with_claude(text)
                text_drugs = text_result.get("drugs", []) or []
                if text_drugs:
                    avg_conf = compute_avg_confidence(text_drugs, confidence_for_engine(engine))
                    return DrugExtractionResponse(
                        success=True, text=text, engine=f"vision_failed+{engine}",
                        drugs=text_drugs,
                        prescription_date=text_result.get("prescription_date"),
                        doctor_name=text_result.get("doctor_name"),
                        hospital_name=text_result.get("hospital_name"),
                        avg_confidence=avg_conf,
                    )
            return DrugExtractionResponse(
                success=True, text="", engine="vision_failed",
                drugs=[], avg_confidence=0,
            )

        # Vision succeeded — return its result
        avg_conf = compute_avg_confidence(vision_drugs, 85)
        return DrugExtractionResponse(
            success=True, text="", engine="claude_vision",
            drugs=vision_drugs,
            prescription_date=vision_result.get("prescription_date"),
            doctor_name=vision_result.get("doctor_name"),
            hospital_name=vision_result.get("hospital_name"),
            avg_confidence=avg_conf,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Drug extraction failed")
        return DrugExtractionResponse(
            success=False, text="", engine="none",
            drugs=[], avg_confidence=0, error=str(e),
        )
# ================================================================
# ADD THIS FUNCTION to main.py 
# Place it right before the _do_lab_report function (line ~1328)
# ================================================================

LAB_REPORT_VISION_PROMPT = """You are a clinical lab technician looking at a photograph of an Indian lab report (AMPATH, SRL, Thyrocare, Metropolis, Dr. Lal Path Labs, Apollo, Vijaya, etc.).

The image may have watermarks, stamps, or diagonal text overlaid on it. READ THROUGH the watermarks carefully — the actual test values are printed underneath. Do not let watermark text confuse you.

# YOUR TASK

Parse this lab report into STRUCTURED JSON. A SINGLE upload may contain MULTIPLE PANELS — split them into separate panel objects.

# DATE EXTRACTION (CRITICAL)

Find `report_date`:
- Look for "Reported on:", "Report Date:", "Date:", "Sample collected:", "Tested on:", "Approved on:"
- Indian format is often DD/MM/YYYY (e.g., "29/06/2026" = June 29, 2026)
- Normalize to YYYY-MM-DD format
- If no date found, use null

# READING VALUES THROUGH WATERMARKS

Indian lab reports often have diagonal watermarks (lab logo, "CONFIDENTIAL", lab name).
- Focus on the RESULT column — values are typically in the middle column
- Reference ranges are in the right column
- If a value appears partially obscured, use context clues (reference range, units) to determine the correct value
- For example: if reference range is 0.50-0.90 and you can see "0.4" with an "L" flag, the value is 0.4
- The "L" suffix means Low, "H" means High — these are FLAGS not part of the value

# CATEGORIES

Auto-categorize each panel into ONE of:
- "Blood" — CBC, LFT, RFT/KFT, Lipid Panel, HbA1c, Thyroid, Glucose, Vitamin D/B12, Iron, Electrolytes
- "Urine" — Urinalysis, Microalbumin, 24-hr protein, Urine culture
- "Imaging" — X-Ray, CT, MRI, Ultrasound, Mammogram, Echo
- "Pathology" — Biopsy, FNAC, Cytology, Histopathology
- "Cardiac" — ECG, 2D Echo, TMT, Holter
- "Other"

# OUTPUT FORMAT

Return ONLY a JSON object. No prose, no markdown fences:
{
  "lab_name": "name or null",
  "report_date": "YYYY-MM-DD or null",
  "patient_name": "name or null",
  "panels": [
    {
      "panel_name": "Renal Function Tests",
      "category": "Blood",
      "tests": [
        {
          "name": "Creatinine",
          "value": 0.4,
          "unit": "mg/dL",
          "normal_range": "0.50-0.90",
          "flag": "low"
        }
      ]
    }
  ]
}

# RULES

1. **value**: Number for numeric (0.4, 142.4). String for qualitative ("Yellow", "Trace"). Null ONLY if completely unreadable after careful inspection.
2. **unit**: Exactly as printed. Null if no unit.
3. **normal_range**: Extract from report. Null if not present.
4. **flag**: "low" | "high" | "critical_low" | "critical_high" | "abnormal" | "normal" | null
5. Extract EVERY test you see. Don't skip any.
6. Look carefully through watermarks — most values ARE readable if you focus on the result column.

Now carefully read the lab report image and return the JSON object:"""


def parse_lab_report_with_claude_vision(image_bytes: bytes) -> dict:
    """
    Send lab report image DIRECTLY to Claude Vision.
    Much better than OCR for watermarked Indian lab reports.
    Claude can read through watermarks and understand document context.
    Returns same shape as parse_lab_report_with_claude().
    """
    if not ANTHROPIC_API_KEY or not image_bytes:
        return {"panels": [], "abnormal_findings": [], "lab_name": None, 
                "report_date": None, "patient_name": None}

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 6144,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": LAB_REPORT_VISION_PROMPT},
            ],
        }],
    }

    text = ""
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=payload, timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        content_blocks = data.get("content", [])
        text = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        ).strip()

        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        result = json.loads(text)
        if not isinstance(result, dict):
            return {"panels": [], "abnormal_findings": [], "lab_name": None,
                    "report_date": None, "patient_name": None}

        result["report_date"] = normalize_date(result.get("report_date"))
        result = post_process_lab_report(result)

        n_panels = len(result.get("panels", []))
        n_tests = sum(len(p.get("tests", [])) for p in result.get("panels", []))
        n_abnormal = len(result.get("abnormal_findings", []))
        logger.info("Vision lab parsed: %d panels, %d tests, %d abnormal | date: %s",
                    n_panels, n_tests, n_abnormal, result.get("report_date"))

        return result

    except json.JSONDecodeError as e:
        logger.error("Claude vision returned invalid JSON for lab: %s | text: %s", e, text[:300])
        return {"panels": [], "abnormal_findings": [], "lab_name": None,
                "report_date": None, "patient_name": None}
    except Exception as e:
        logger.error("Claude vision lab parsing failed: %s", e)
        return {"panels": [], "abnormal_findings": [], "lab_name": None,
                "report_date": None, "patient_name": None}


# ================================================================
#  _do_lab_report function with this version
# ================================================================

async def _do_lab_report(file: UploadFile) -> LabReportResponse:
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        is_pdf = detect_pdf(file, file_bytes)

        # =================================================================
        # PDF PATH: text extraction works well for PDFs — keep as-is
        # =================================================================
        if is_pdf:
            text, engine = get_text_from_upload(file, file_bytes)
            if engine == "pdf_failed" or not text:
                return LabReportResponse(
                    success=False, text="", engine=engine,
                    panels=[], abnormal_findings=[],
                    avg_confidence=0,
                    error="Could not extract any text from PDF",
                )
            parsed = parse_lab_report_with_claude(text)
            return LabReportResponse(
                success=True,
                text=text, engine=engine,
                lab_name=parsed.get("lab_name"),
                report_date=parsed.get("report_date"),
                patient_name=parsed.get("patient_name"),
                panels=parsed.get("panels", []),
                abnormal_findings=parsed.get("abnormal_findings", []),
                avg_confidence=confidence_for_engine(engine),
            )

        # =================================================================
        # IMAGE PATH: Use Claude Vision directly
        # Much better than OCR for watermarked Indian lab reports
        # Claude reads through watermarks and understands document context
        # =================================================================
        logger.info("Image lab report — using Claude Vision directly")
        parsed = parse_lab_report_with_claude_vision(file_bytes)

        if not parsed.get("panels"):
            # Vision failed — fall back to OCR pipeline
            logger.warning("Claude vision returned no panels — falling back to OCR")
            text, engine = extract_text_with_fallback(file_bytes)
            if text:
                parsed = parse_lab_report_with_claude(text)
                if parsed.get("panels"):
                    return LabReportResponse(
                        success=True,
                        text=text, engine=f"vision_failed+{engine}",
                        lab_name=parsed.get("lab_name"),
                        report_date=parsed.get("report_date"),
                        patient_name=parsed.get("patient_name"),
                        panels=parsed.get("panels", []),
                        abnormal_findings=parsed.get("abnormal_findings", []),
                        avg_confidence=confidence_for_engine(engine),
                    )
            return LabReportResponse(
                success=False, text="", engine="vision_failed",
                panels=[], abnormal_findings=[],
                avg_confidence=0,
                error="Could not extract any data from image",
            )

        return LabReportResponse(
            success=True,
            text="", engine="claude_vision",
            lab_name=parsed.get("lab_name"),
            report_date=parsed.get("report_date"),
            patient_name=parsed.get("patient_name"),
            panels=parsed.get("panels", []),
            abnormal_findings=parsed.get("abnormal_findings", []),
            avg_confidence=92,  # Claude Vision is highly accurate
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Lab report parsing failed")
        return LabReportResponse(
            success=False, text="", engine="none",
            panels=[], abnormal_findings=[],
            avg_confidence=0, error=str(e),
        )

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "MedRecord OCR",
        "status": "running",
        "version": "1.9.2",
        "google_vision_configured": bool(GOOGLE_VISION_KEY),
        "claude_configured": bool(ANTHROPIC_API_KEY),
        "supported_formats": ["JPEG", "PNG", "PDF (digital and scanned)"],
        "features": {
            "handwriting_auto_detect": True,
            "claude_vision_fallback": True,
            "multi_panel_labs": True,
            "duplicate_detection": False,
        },
        "endpoints": [
            "/health", "/ocr", "/ocr/lab", "/ocr/prescription",
            "/ocr/report", "/extract-drugs", "/extract_drugs", "/docs",
        ],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    return await _do_ocr(file)


@app.post("/ocr/lab", response_model=OCRResponse)
async def ocr_lab_endpoint(file: UploadFile = File(...)):
    return await _do_ocr(file)


@app.post("/extract-drugs", response_model=DrugExtractionResponse)
async def extract_drugs_endpoint(file: UploadFile = File(...)):
    return await _do_extract_drugs(file)


@app.post("/extract_drugs", response_model=DrugExtractionResponse)
async def extract_drugs_underscore_endpoint(file: UploadFile = File(...)):
    return await _do_extract_drugs(file)


@app.post("/ocr/prescription", response_model=DrugExtractionResponse)
async def ocr_prescription_endpoint(file: UploadFile = File(...)):
    return await _do_extract_drugs(file)


@app.post("/ocr/report", response_model=LabReportResponse)
async def ocr_report_endpoint(file: UploadFile = File(...)):
    return await _do_lab_report(file)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
