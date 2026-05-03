"""
MedRecord OCR Microservice
FastAPI + Google Vision (REST) + EasyOCR fallback + Claude Haiku 4.5 drug extraction

v1.3.0 — App compatibility layer:
  - Drug objects now expose BOTH legacy & new field names so React Native PrescriptionsScreen.js works:
      * drug_name (app expects)        + name (backend native)
      * type: hospital | outpatient    + source: hospital | current
  - Adds avg_confidence so the alert displays a real number
  - Carries forward v1.2.0 deterministic IV->hospital classification
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
    logger.warning("ANTHROPIC_API_KEY not set — Claude drug extraction will be skipped")

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
    description="OCR + drug extraction for lab reports and prescriptions",
    version="1.3.0",
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
    avg_confidence: int = 0  # so the app's alert shows a real number
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


# ------------------------------------------------------------------
# Drug extraction — Claude + deterministic post-processing
# ------------------------------------------------------------------
DRUG_EXTRACTION_PROMPT = """You are a clinical pharmacist extracting EVERY medication from an Indian hospital discharge summary or prescription. Do not skip ANY drug, even if it appears only once or in a list.

OCR TEXT:
---
{ocr_text}
---

# YOUR TASK

Extract ALL medications. Indian discharge summaries have TWO distinct medication sections that you must scan:

## Section 1: HOSPITAL MEDICATIONS (drugs given DURING admission)
Look under headings like:
- "Course in the Hospital"
- "Medication" / "Medications Given"
- "Treatment Given"
- "During Stay"
- "Inpatient Medications"

These are typically:
- IV / Intravenous infusions
- Injections (Inj., INJ)
- Drugs with route "IV", "infusion", "subcutaneous"
- Chemo drugs (Docetaxel, Carboplatin, Phesgo, etc.)
- Premedications (Ondansetron, Pantoprazole, Pheniramine, Dexamethasone, Fosaprepitant, etc.)

For these, set "source": "hospital".

## Section 2: DISCHARGE / TAKE-HOME MEDICATIONS
Look under headings like:
- "Medication Advise" / "Discharge Medication"
- "Take Home Medications"
- "Advise at the time of discharge"
- "TAB." / "CAP." / "SYP." prefixes (oral forms)

These are oral pills with dosing schedules like "1-0-0", "1-1-1", "BD", "TDS", "HS", "SOS".

For these, set "source": "current".

# OUTPUT FORMAT

Return ONLY a JSON array. No prose, no markdown fences. Each drug:
{{
  "name": "drug name in TitleCase (generic or brand as written)",
  "dosage": "e.g. '500mg', '8 mg', '150 ml' or null",
  "frequency": "e.g. 'BD', 'TDS', '1-0-0', 'once daily', 'every 8h' or null",
  "duration": "e.g. '5 days', '1 month' or null",
  "route": "oral | IV | subcutaneous | topical | inhalation",
  "source": "hospital | current"
}}

# INDIAN ABBREVIATIONS YOU MUST KNOW
- BD / BID = twice daily
- TDS / TID = three times daily
- QID = four times daily
- HS = at bedtime
- SOS = as needed
- 1-0-0 = morning only
- 1-1-1 = morning, afternoon, night
- 1-0-1 = morning and night
- 0-0-1 = night only
- TAB = tablet (oral)
- CAP = capsule (oral)
- INJ = injection (IV unless stated)
- SYP = syrup (oral)

# CRITICAL RULES

1. EXTRACT EVERY DRUG. Do not summarize, do not skip duplicates, do not merge similar names.
2. If a drug is listed under chemo/IV section AND has "infusion" or "Inj" — it is "hospital" source.
3. If a drug starts with "TAB."/"CAP."/"SYP." — it is "current" source, oral route.
4. If a drug has dosing notation like "1-0-0" or "BD" — it is "current" source.
5. If unsure about source, default to "current".
6. Return [] only if absolutely no medications found.

Now extract the medications as a JSON array:"""


def extract_drugs_with_claude(ocr_text: str) -> list:
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — skipping drug extraction")
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

        # Strip markdown fences if Claude wrapped JSON
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        # Find the JSON array even if Claude added stray text
        json_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        drugs = json.loads(text)
        if not isinstance(drugs, list):
            logger.warning("Claude returned non-list: %s", type(drugs))
            return []

        drugs = post_process_drugs(drugs)
        # Add app-compatible fields AFTER post-processing
        drugs = add_app_compat_fields(drugs)

        logger.info(
            "Claude extracted %d drugs (%d hospital, %d current)",
            len(drugs),
            sum(1 for d in drugs if d.get("source") == "hospital"),
            sum(1 for d in drugs if d.get("source") == "current"),
        )
        return drugs
    except json.JSONDecodeError as e:
        logger.error("Claude returned invalid JSON: %s | text was: %s", e, text[:500])
        return []
    except Exception as e:
        logger.error("Claude drug extraction failed: %s", e)
        return []


def post_process_drugs(drugs: list) -> list:
    """
    Deterministic rules applied AFTER Claude:
      - IV / infusion / subcutaneous route -> 'hospital'
      - Drug name starting with 'INJ' -> 'hospital'
      - Drug name starting with 'TAB'/'CAP'/'SYP' -> 'current', oral
    """
    cleaned = []
    for d in drugs:
        if not isinstance(d, dict):
            continue
        if not d.get("name"):
            continue

        name = str(d.get("name", "")).strip()
        route = str(d.get("route", "") or "").lower()

        is_iv_route = any(
            kw in route for kw in ["iv", "intravenous", "infusion", "subcutaneous", "inj"]
        )
        is_inj_name = bool(re.match(r"^\s*inj\.?\b", name, re.IGNORECASE))
        if is_iv_route or is_inj_name:
            d["source"] = "hospital"

        is_oral_prefix = bool(re.match(r"^\s*(tab|cap|syp|syr)\.?\b", name, re.IGNORECASE))
        if is_oral_prefix:
            d["source"] = "current"
            if not d.get("route"):
                d["route"] = "oral"

        if not d.get("source"):
            d["source"] = "current"

        cleaned.append(d)
    return cleaned


def add_app_compat_fields(drugs: list) -> list:
    """
    Add field aliases the React Native app expects:
      - drug_name  (mirror of 'name')
      - type       ('hospital' for hospital source, 'outpatient' for current)
    Keeps original fields too so /docs and any future client still works.
    """
    for d in drugs:
        # Add drug_name alias
        if "name" in d and "drug_name" not in d:
            d["drug_name"] = d["name"]

        # Map source -> type for app filtering
        # App expects: 'hospital' | 'discharge' | 'outpatient'
        source = d.get("source", "current")
        if source == "hospital":
            d["type"] = "hospital"
        else:  # 'current' or anything else
            d["type"] = "outpatient"

        # Provide a category default if missing
        if not d.get("category"):
            d["category"] = "Medication"

    return drugs


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "MedRecord OCR",
        "status": "running",
        "version": "1.3.0",
        "google_vision_configured": bool(GOOGLE_VISION_KEY),
        "claude_configured": bool(ANTHROPIC_API_KEY),
        "endpoints": ["/health", "/ocr", "/extract-drugs", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
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
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")
        text, engine = extract_text_with_fallback(image_bytes)
        drugs = extract_drugs_with_claude(text)
        # Confidence is a heuristic for now: 95 if Google Vision worked, 75 if EasyOCR fallback
        avg_conf = 95 if engine == "google_vision" else 75
        return DrugExtractionResponse(
            success=True,
            text=text,
            engine=engine,
            drugs=drugs,
            avg_confidence=avg_conf,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Drug extraction failed")
        return DrugExtractionResponse(
            success=False,
            text="",
            engine="none",
            drugs=[],
            avg_confidence=0,
            error=str(e),
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)