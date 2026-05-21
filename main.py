"""
MedRecord OCR Microservice
FastAPI + Google Vision (REST) + EasyOCR fallback + Claude Haiku 4.5

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
    version="1.8.0",
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
DRUG_VISION_PROMPT = """You are a clinical pharmacist looking at a photograph of an Indian doctor's prescription or hospital discharge summary. The handwriting may be messy. Read it carefully, drug by drug.

# YOUR TASK

Return a JSON object with two parts:

## Part 1: METADATA
- prescription_date: Visit/consultation/discharge date in YYYY-MM-DD format. Look in printed header AND handwritten notes.
- doctor_name: Prescribing doctor's name (e.g., "Dr. Rajesh Kumar"). Null if not present.
- hospital_name: Hospital or clinic name. Null if not present.

## Part 2: DRUGS — extract EVERY medication you see
Indian prescriptions usually number drugs (1, 2, 3...) often scattered across the page.
- Look at ALL columns, not just top-to-bottom
- Drug names usually start with "Tab.", "Cap.", "Syp.", "Inj."
- Common Indian brand names: Deplatt, Ecospirin, Telma, Atorva, Metoprolol, Pan, Crocin, Dolo, Glycomet, Amlong, Concor, Eptus, Dytor, Hyponat, L-Montus, Valentas
- Indian dosing notation: 1-0-0 (morning only), 1-0-1 (morning+night), 1-1-1 (3x/day), BD/TDS/QID, HS (bedtime), SOS (as needed), PO OD (per oral once daily)
- If a drug name is partially illegible, give your best phonetic guess and lower the confidence

## CONFIDENCE
- For each drug, include "confidence": "high", "medium", or "low" based on how clearly you could read it
- "high" = name and dose clearly legible
- "medium" = name OR dose required interpretation
- "low" = significant guesswork on name or dose

# OUTPUT FORMAT

Return ONLY JSON, no prose, no markdown fences:
{
  "prescription_date": "YYYY-MM-DD or null",
  "doctor_name": "string or null",
  "hospital_name": "string or null",
  "drugs": [
    {
      "name": "drug name as written",
      "dosage": "e.g. '500mg' or null",
      "frequency": "BD | TDS | 1-0-1 | 1-1-1 | HS | SOS | OD | etc or null",
      "duration": "e.g. '5 days' or null",
      "route": "oral | IV | subcutaneous | topical | inhalation | null",
      "source": "current | hospital",
      "confidence": "high | medium | low"
    }
  ]
}

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

        # ---- Stage 1: cheap OCR + Haiku text pipeline ----
        text, engine = get_text_from_upload(file, file_bytes)
        if engine == "pdf_failed" or not text:
            # For PDFs we can't do vision fallback (would need rasterization which
            # we already attempted in extract_text_from_pdf). Return failure.
            return DrugExtractionResponse(
                success=False, text="", engine=engine,
                drugs=[], avg_confidence=0,
                error="Could not extract any text from upload",
            )

        text_result = extract_drugs_with_claude(text)
        text_drugs = text_result.get("drugs", []) or []

        # ---- Stage 2: cascade decision ----
        # If the cheap path found enough drugs, return that.
        # If <2 drugs AND we have a real image (not PDF), try Claude vision.
        if len(text_drugs) >= 2 or is_pdf:
            avg_conf = compute_avg_confidence(text_drugs, confidence_for_engine(engine))
            return DrugExtractionResponse(
                success=True, text=text, engine=engine,
                drugs=text_drugs,
                prescription_date=text_result.get("prescription_date"),
                doctor_name=text_result.get("doctor_name"),
                hospital_name=text_result.get("hospital_name"),
                avg_confidence=avg_conf,
            )

        logger.info("Cheap path returned %d drug(s) — falling back to Claude vision", len(text_drugs))

        # ---- Stage 3: Claude vision fallback (handwriting!) ----
        vision_result = extract_drugs_with_claude_vision(file_bytes)
        vision_drugs = vision_result.get("drugs", []) or []

        # If vision also returned nothing meaningful, prefer whichever found more
        if not vision_drugs and not text_drugs:
            return DrugExtractionResponse(
                success=True, text=text,
                engine=f"{engine}+vision_failed",
                drugs=[],
                avg_confidence=0,
            )

        if len(vision_drugs) >= len(text_drugs):
            # Vision did better — use its result, blend metadata if missing
            final_drugs = vision_drugs
            final_date = vision_result.get("prescription_date") or text_result.get("prescription_date")
            final_doctor = vision_result.get("doctor_name") or text_result.get("doctor_name")
            final_hospital = vision_result.get("hospital_name") or text_result.get("hospital_name")
            # Vision engine confidence sits between cheap OCR and Google Vision
            avg_conf = compute_avg_confidence(final_drugs, 85)
            return DrugExtractionResponse(
                success=True, text=text,
                engine=f"{engine}+vision",
                drugs=final_drugs,
                prescription_date=final_date,
                doctor_name=final_doctor,
                hospital_name=final_hospital,
                avg_confidence=avg_conf,
            )

        # Cheap path returned more — keep it
        avg_conf = compute_avg_confidence(text_drugs, confidence_for_engine(engine))
        return DrugExtractionResponse(
            success=True, text=text, engine=engine,
            drugs=text_drugs,
            prescription_date=text_result.get("prescription_date"),
            doctor_name=text_result.get("doctor_name"),
            hospital_name=text_result.get("hospital_name"),
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


async def _do_lab_report(file: UploadFile) -> LabReportResponse:
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")
        text, engine = get_text_from_upload(file, file_bytes)
        if engine == "pdf_failed" or not text:
            return LabReportResponse(
                success=False, text="", engine=engine,
                panels=[], abnormal_findings=[],
                avg_confidence=0,
                error="Could not extract any text from upload",
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
        "version": "1.8.0",
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
