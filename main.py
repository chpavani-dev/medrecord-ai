"""
MedRecord OCR Microservice
FastAPI + Google Vision (REST) + EasyOCR fallback + Claude Haiku 4.5

v1.5.0 — Lab report intelligence:
  - NEW: /ocr/report endpoint for multi-panel lab parsing
  - Auto-categorization: Blood, Urine, Imaging, Pathology, Cardiac, Other
  - Test name normalization (S. Creatinine, Creat, Creatinine -> "Creatinine")
  - Abnormal value flagging using report's normal range OR standard ranges
  - Returns abnormal_findings array for Trends auto-promotion

  Carries forward:
  - v1.4.0: /ocr/prescription, /ocr/lab, /extract_drugs aliases
  - v1.3.0: drug_name, type, avg_confidence app-compat fields
  - v1.2.0: deterministic IV->hospital classification
"""

import os
import base64
import json
import logging
import re
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
# Standard 9 trends metrics (your existing Tier 1)
# ------------------------------------------------------------------
STANDARD_METRICS = {
    "hba1c", "glucose", "hemoglobin", "haemoglobin", "tsh",
    "cholesterol", "ldl", "hdl", "triglycerides", "creatinine",
}

# ------------------------------------------------------------------
# Test name normalization map — variants -> canonical name
# ------------------------------------------------------------------
TEST_NAME_NORMALIZATION = {
    # Format: "lowercase_match_key": "Canonical Display Name"
    "creatinine": "Creatinine",
    "s. creatinine": "Creatinine",
    "serum creatinine": "Creatinine",
    "creat": "Creatinine",
    "hemoglobin": "Hemoglobin",
    "haemoglobin": "Hemoglobin",
    "hgb": "Hemoglobin",
    "hb": "Hemoglobin",
    "hba1c": "HbA1c",
    "glycated hemoglobin": "HbA1c",
    "glycosylated hemoglobin": "HbA1c",
    "glucose": "Glucose",
    "fasting glucose": "Fasting Glucose",
    "fbs": "Fasting Glucose",
    "ppbs": "Post Prandial Glucose",
    "post prandial glucose": "Post Prandial Glucose",
    "rbs": "Random Glucose",
    "tsh": "TSH",
    "thyroid stimulating hormone": "TSH",
    "t3": "T3",
    "t4": "T4",
    "free t3": "Free T3",
    "free t4": "Free T4",
    "cholesterol": "Total Cholesterol",
    "total cholesterol": "Total Cholesterol",
    "ldl": "LDL Cholesterol",
    "ldl cholesterol": "LDL Cholesterol",
    "hdl": "HDL Cholesterol",
    "hdl cholesterol": "HDL Cholesterol",
    "triglycerides": "Triglycerides",
    "tg": "Triglycerides",
    "uric acid": "Uric Acid",
    "vitamin d": "Vitamin D",
    "25 oh vitamin d": "Vitamin D",
    "25-oh vitamin d": "Vitamin D",
    "vitamin b12": "Vitamin B12",
    "b12": "Vitamin B12",
    "wbc": "WBC",
    "white blood cells": "WBC",
    "rbc": "RBC",
    "red blood cells": "RBC",
    "platelets": "Platelets",
    "plt": "Platelets",
    "esr": "ESR",
    "crp": "CRP",
    "sgot": "SGOT (AST)",
    "ast": "SGOT (AST)",
    "sgpt": "SGPT (ALT)",
    "alt": "SGPT (ALT)",
    "bilirubin": "Bilirubin Total",
    "total bilirubin": "Bilirubin Total",
    "urea": "Urea",
    "blood urea": "Urea",
    "bun": "BUN",
    "sodium": "Sodium",
    "na": "Sodium",
    "potassium": "Potassium",
    "k": "Potassium",
}


def normalize_test_name(raw_name: str) -> str:
    """Normalize a test name to canonical form. Falls back to TitleCase if unknown."""
    if not raw_name:
        return raw_name
    key = raw_name.strip().lower()
    # Strip leading "S." (serum) or "P." (plasma) prefixes
    key = re.sub(r"^[sp]\.\s*", "", key)
    if key in TEST_NAME_NORMALIZATION:
        return TEST_NAME_NORMALIZATION[key]
    # Fallback: TitleCase the original
    return raw_name.strip().title()


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
    description="OCR + AI parsing for prescriptions and lab reports",
    version="1.5.0",
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
        logger.warning("Google Vision returned no 'responses' field")
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
    try:
        text = extract_text_google_vision(image_bytes)
        if text and text.strip():
            logger.info("Google Vision succeeded (%d chars)", len(text))
            return text, "google_vision"
        logger.warning("Google Vision returned empty text — falling back to EasyOCR")
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
# DRUG EXTRACTION (unchanged from v1.4.0)
# ==================================================================
DRUG_EXTRACTION_PROMPT = """You are a clinical pharmacist extracting EVERY medication from an Indian hospital discharge summary or prescription. Do not skip ANY drug.

OCR TEXT:
---
{ocr_text}
---

# YOUR TASK

Extract ALL medications. Indian discharge summaries have TWO sections:

## Section 1: HOSPITAL MEDICATIONS (drugs given during admission)
- IV / Intravenous infusions, Injections (Inj.)
- Chemo drugs (Docetaxel, Carboplatin, Phesgo, etc.)
- Premedications (Ondansetron, Pantoprazole, Pheniramine, Dexamethasone, Fosaprepitant, etc.)
For these, set "source": "hospital".

## Section 2: DISCHARGE / TAKE-HOME MEDICATIONS
- TAB./CAP./SYP. prefixes, oral pills
- Indian dosing: 1-0-0, 1-1-1, BD, TDS, HS, SOS
For these, set "source": "current".

# OUTPUT FORMAT
Return ONLY a JSON array. No prose, no markdown fences:
{{
  "name": "drug name",
  "dosage": "e.g. '500mg' or null",
  "frequency": "e.g. 'BD', '1-0-0' or null",
  "duration": "e.g. '5 days' or null",
  "route": "oral | IV | subcutaneous | topical | inhalation",
  "source": "hospital | current"
}}

# ABBREVIATIONS
BD=twice daily, TDS=three times daily, QID=four times, HS=bedtime, SOS=as needed
1-0-0=morning, 1-1-1=morning/afternoon/night, 1-0-1=morning/night

Now extract the medications as a JSON array:"""


def extract_drugs_with_claude(ocr_text: str) -> list:
    if not ANTHROPIC_API_KEY:
        return []
    if not ocr_text or not ocr_text.strip():
        return []

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
            headers=headers,
            json=payload,
            timeout=60,
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

        json_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        drugs = json.loads(text)
        if not isinstance(drugs, list):
            return []

        drugs = post_process_drugs(drugs)
        drugs = add_drug_app_compat_fields(drugs)
        logger.info("Drugs: %d total (%d hospital, %d current)",
                    len(drugs),
                    sum(1 for d in drugs if d.get("source") == "hospital"),
                    sum(1 for d in drugs if d.get("source") == "current"))
        return drugs
    except json.JSONDecodeError as e:
        logger.error("Claude returned invalid JSON for drugs: %s", e)
        return []
    except Exception as e:
        logger.error("Claude drug extraction failed: %s", e)
        return []


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
# LAB REPORT EXTRACTION (NEW in v1.5.0)
# ==================================================================
LAB_REPORT_PROMPT = """You are a clinical lab technician parsing a lab report from an Indian diagnostic lab (SRL, Thyrocare, Metropolis, Dr. Lal Path Labs, Apollo, Vijaya, etc.).

OCR TEXT:
---
{ocr_text}
---

# YOUR TASK

Parse this lab report and extract EVERY test result. Return STRUCTURED JSON.

A SINGLE upload may contain MULTIPLE PANELS — split them into separate panel objects.

## CATEGORIES TO USE

Auto-categorize each panel into ONE of:
- "Blood" — CBC, LFT, RFT/KFT, Lipid Panel, HbA1c, Thyroid (TSH/T3/T4), Glucose, Vitamin D/B12, Iron Studies, Electrolytes
- "Urine" — Urinalysis, Microalbumin, 24-hr protein, Urine culture
- "Imaging" — X-Ray, CT, MRI, Ultrasound, Mammogram, Echo
- "Pathology" — Biopsy, FNAC, Cytology, Histopathology
- "Cardiac" — ECG, 2D Echo, TMT, Holter
- "Other" — anything else

## DETECTING PANELS

A panel is a logical grouping of related tests. Common Indian lab panels:
- "Complete Blood Count" / "CBC" / "Hemogram" → Blood (Hemoglobin, WBC, RBC, Platelets, MCV, MCH, etc.)
- "Lipid Profile" / "Lipid Panel" → Blood (Cholesterol, LDL, HDL, Triglycerides)
- "Liver Function Test" / "LFT" → Blood (Bilirubin, SGOT, SGPT, Albumin)
- "Renal Function Test" / "RFT" / "KFT" → Blood (Urea, Creatinine, BUN, Uric Acid)
- "Thyroid Profile" → Blood (TSH, T3, T4)
- "HbA1c" → Blood (HbA1c, often standalone)
- "Glucose" / "FBS" / "PPBS" → Blood
- "Urinalysis" / "Urine Routine" → Urine
- "Vitamin D" / "Vitamin B12" → Blood

If panel name is unclear, infer from the tests present (e.g., Hb + WBC + Platelets = "Complete Blood Count").

## OUTPUT FORMAT

Return ONLY a JSON object (NOT array). No prose, no markdown fences:
{{
  "lab_name": "name of lab/diagnostic center or null",
  "report_date": "YYYY-MM-DD or null",
  "patient_name": "patient name or null",
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

## RULES FOR EACH TEST

1. **value**: Use a number for numeric results (12.5, 7800, 0.8). Use a string for qualitative (e.g., "Yellow", "Trace", "Negative", "Positive"). Use null if missing.
2. **unit**: Extract exactly as printed (g/dL, mg/dL, /cumm, %, IU/L, ng/mL). Null if no unit.
3. **normal_range**: Extract from the report (e.g., "13.0-17.0", "<200", ">40"). Null if not in report.
4. **flag**: One of:
   - "low" — value below normal range
   - "high" — value above normal range
   - "critical_low" — dangerously low (e.g., Hb < 7, K < 2.5, Glucose < 50)
   - "critical_high" — dangerously high (e.g., K > 6, Glucose > 400, Creat > 5)
   - "abnormal" — for qualitative results that aren't normal (e.g., Protein "Trace" when normal is "Negative")
   - "normal" — within range
   - null — cannot determine
5. **name**: Use the test name as printed. Don't normalize — backend handles that.

## CRITICAL RULES

- Extract EVERY test you see, even ones you don't recognize. Don't skip.
- If you can't determine the flag, use null — don't guess.
- For Indian reports, "S." prefix means "Serum" (e.g., "S. Creatinine" = Creatinine in serum) — keep the name as-is.
- Lab name is often in the header/footer (e.g., "SRL Diagnostics", "Thyrocare", "Dr. Lal Path Labs").
- Report date format is often DD/MM/YYYY in India — convert to YYYY-MM-DD.
- If multiple dates exist (collection, report), prefer the report date.

Now parse the lab report and return the JSON object:"""


# Standard reference ranges (used as fallback when report doesn't include them)
STANDARD_REFERENCE_RANGES = {
    # Lab values: (low, high, unit, critical_low, critical_high)
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
    """Determine flag (normal/low/high/critical) — uses report's normal range if available, else standard ranges."""
    if value is None:
        return None
    # Skip flagging for non-numeric values (we trust Claude's flag for those)
    if not isinstance(value, (int, float)):
        return None

    # First try standard reference ranges (more reliable than parsing arbitrary range strings)
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

    # Fallback: try to parse the report's normal_range string
    if normal_range:
        rng = normal_range.strip()
        # Handle "X-Y" format
        m = re.match(r"^([\d.]+)\s*[-–]\s*([\d.]+)", rng)
        if m:
            low, high = float(m.group(1)), float(m.group(2))
            if value < low:
                return "low"
            if value > high:
                return "high"
            return "normal"
        # Handle "<X" format (upper limit only)
        m = re.match(r"^<\s*([\d.]+)", rng)
        if m:
            high = float(m.group(1))
            return "high" if value > high else "normal"
        # Handle ">X" format (lower limit only)
        m = re.match(r"^>\s*([\d.]+)", rng)
        if m:
            low = float(m.group(1))
            return "low" if value < low else "normal"

    return None


def parse_lab_report_with_claude(ocr_text: str) -> dict:
    """Parse lab report into structured panels + tests with abnormal_findings."""
    if not ANTHROPIC_API_KEY:
        return {"panels": [], "abnormal_findings": []}
    if not ocr_text or not ocr_text.strip():
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
            headers=headers,
            json=payload,
            timeout=90,
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

        # Find the JSON object (Claude returns object for lab reports, not array)
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        result = json.loads(text)
        if not isinstance(result, dict):
            logger.warning("Claude returned non-dict for lab: %s", type(result))
            return {"panels": [], "abnormal_findings": []}

        # Post-process: normalize names, re-flag values, build abnormal_findings
        result = post_process_lab_report(result)
        return result
    except json.JSONDecodeError as e:
        logger.error("Claude returned invalid JSON for lab: %s | text: %s", e, text[:500])
        return {"panels": [], "abnormal_findings": []}
    except Exception as e:
        logger.error("Claude lab parsing failed: %s", e)
        return {"panels": [], "abnormal_findings": []}


def post_process_lab_report(result: dict) -> dict:
    """Normalize test names, refine flags, build abnormal_findings list."""
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
            t["original_name"] = raw_name  # keep original for reference

            # Mark if it's a standard Tier 1 metric
            t["is_standard_metric"] = canonical.lower() in STANDARD_METRICS

            # Recompute flag using our deterministic ranges (trust this over Claude's flag for numeric values)
            value = t.get("value")
            normal_range = t.get("normal_range") or ""
            our_flag = determine_flag(canonical, value, normal_range)
            if our_flag is not None:
                t["flag"] = our_flag

            # Build abnormal_findings entry
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
    logger.info("Lab parsed: %d panels, %d tests, %d abnormal", n_panels, n_tests, n_abnormal)

    return result


# ------------------------------------------------------------------
# Shared business logic
# ------------------------------------------------------------------
async def _do_ocr(file: UploadFile) -> OCRResponse:
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


async def _do_extract_drugs(file: UploadFile) -> DrugExtractionResponse:
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")
        text, engine = extract_text_with_fallback(image_bytes)
        drugs = extract_drugs_with_claude(text)
        avg_conf = 95 if engine == "google_vision" else 75
        return DrugExtractionResponse(
            success=True, text=text, engine=engine,
            drugs=drugs, avg_confidence=avg_conf,
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
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")
        text, engine = extract_text_with_fallback(image_bytes)
        parsed = parse_lab_report_with_claude(text)
        avg_conf = 95 if engine == "google_vision" else 75
        return LabReportResponse(
            success=True,
            text=text,
            engine=engine,
            lab_name=parsed.get("lab_name"),
            report_date=parsed.get("report_date"),
            patient_name=parsed.get("patient_name"),
            panels=parsed.get("panels", []),
            abnormal_findings=parsed.get("abnormal_findings", []),
            avg_confidence=avg_conf,
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
        "version": "1.5.0",
        "google_vision_configured": bool(GOOGLE_VISION_KEY),
        "claude_configured": bool(ANTHROPIC_API_KEY),
        "endpoints": [
            "/health",
            "/ocr",
            "/ocr/lab",
            "/ocr/prescription",
            "/ocr/report",
            "/extract-drugs",
            "/extract_drugs",
            "/docs",
        ],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# ----- Plain OCR -----
@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    return await _do_ocr(file)


@app.post("/ocr/lab", response_model=OCRResponse)
async def ocr_lab_endpoint(file: UploadFile = File(...)):
    """Plain OCR alias for lab images."""
    return await _do_ocr(file)


# ----- Drug extraction (prescriptions) -----
@app.post("/extract-drugs", response_model=DrugExtractionResponse)
async def extract_drugs_endpoint(file: UploadFile = File(...)):
    return await _do_extract_drugs(file)


@app.post("/extract_drugs", response_model=DrugExtractionResponse)
async def extract_drugs_underscore_endpoint(file: UploadFile = File(...)):
    return await _do_extract_drugs(file)


@app.post("/ocr/prescription", response_model=DrugExtractionResponse)
async def ocr_prescription_endpoint(file: UploadFile = File(...)):
    """OCR + drug extraction — what the React Native PrescriptionsScreen calls."""
    return await _do_extract_drugs(file)


# ----- Lab report parsing (NEW) -----
@app.post("/ocr/report", response_model=LabReportResponse)
async def ocr_report_endpoint(file: UploadFile = File(...)):
    """
    OCR + multi-panel lab report parsing.
    Returns structured panels with auto-categorization, flagged abnormal values,
    and an abnormal_findings list for Trends auto-promotion.
    """
    return await _do_lab_report(file)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
