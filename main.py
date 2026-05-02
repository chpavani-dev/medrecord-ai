from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import easyocr
from PIL import Image
import io
import json
import re
import os
import base64
import requests
from anthropic import Anthropic
from fuzzywuzzy import fuzz, process
from drug_dictionary import INDIAN_DRUGS, ABBREVIATIONS
from image_processor import enhance_image
from drug_parser import parse_prescription

app = FastAPI(title="MedRecord AI Service", version="1.0.0")

# REPLACE WITH YOUR API KEY
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_VISION_KEY = "AIzaSy..."

claude = Anthropic(api_key=ANTHROPIC_API_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading OCR engine...")
reader = easyocr.Reader(['en', 'hi'], gpu=False)
print("OCR engine ready!")


def freq_to_label(code):
    labels = {
        "OD": "Once daily", "BD": "Twice daily", "TDS": "Three times",
        "QID": "Four times", "HS": "At bedtime", "SOS": "As needed",
        "AC": "Before meals", "PC": "After meals", "STAT": "One-time"
    }
    return labels.get(code, code)


def freq_to_times(code):
    times_map = {
        "OD": ["8:00 AM"],
        "BD": ["8:00 AM", "8:00 PM"],
        "TDS": ["8:00 AM", "2:00 PM", "8:00 PM"],
        "QID": ["8:00 AM", "12:00 PM", "4:00 PM", "8:00 PM"],
        "HS": ["10:00 PM"],
        "SOS": ["As needed"],
        "STAT": ["One-time dose"],
        "AC": ["Before meals"],
        "PC": ["After meals"],
    }
    return times_map.get(code, ["8:00 AM"])


def extract_drugs_with_claude(raw_text):
    try:
        print("\n=== OCR TEXT TO CLAUDE ===")
        print(raw_text[:2000])
        print("=== END ===\n")

        prompt = "You are a medical assistant that extracts medications from prescription text.\n\n"
        prompt += "Below is raw OCR text from a prescription or hospital discharge summary. Extract ALL medications.\n\n"
        prompt += "EXTRACTION RULES:\n"
        prompt += "1. Hospital discharge summaries have TWO medication sections - extract BOTH:\n"
        prompt += "   - Course in Hospital / Given section: IV drugs administered during admission\n"
        prompt += "   - Medication Advise / Discharge medications section: drugs to take at home\n\n"
        prompt += "2. Mark each drug with its source using the type field:\n"
        prompt += "   - hospital = given during hospital stay\n"
        prompt += "   - discharge = take at home after discharge\n"
        prompt += "   - outpatient = regular OPD prescription\n\n"
        prompt += "3. Indian frequency notation:\n"
        prompt += "   - 1-0-0 = OD (morning only)\n"
        prompt += "   - 0-0-1 = HS (night only)\n"
        prompt += "   - 1-0-1 = BD (morning and night)\n"
        prompt += "   - 1-1-1 = TDS (3 times)\n"
        prompt += "   - 1-1-1-1 = QID (4 times)\n"
        prompt += "   - thrice daily / TDS / TID = TDS\n"
        prompt += "   - twice daily / BD / BID = BD\n"
        prompt += "   - once daily / OD / QD = OD\n"
        prompt += "   - at bedtime / HS = HS\n\n"
        prompt += "4. For IV/Injection drugs: frequency = STAT (one-time)\n\n"
        prompt += "5. For each medication return:\n"
        prompt += "   - name: drug name\n"
        prompt += "   - dose: dosage with units\n"
        prompt += "   - frequency: OD/BD/TDS/QID/HS/SOS/STAT\n"
        prompt += "   - duration: number of days (default 30, use 1 for IV)\n"
        prompt += "   - category: Diabetes, Hypertension, Cholesterol, Thyroid, Antibiotic, Pain, Gastro, Heart, Respiratory, Supplement, Psychiatric, Cancer, Chemotherapy, Steroid, Antiemetic, or Other\n"
        prompt += "   - type: hospital, discharge, or outpatient\n"
        prompt += "   - notes: special instructions like before food, IV, subcutaneous\n"
        prompt += "   - route: oral, iv, im, sc, topical, inhaler, eye_drops\n\n"
        prompt += f"OCR TEXT:\n{raw_text}\n\n"
        prompt += 'Respond ONLY with valid JSON, no other text:\n'
        prompt += '{"drugs": [{"name": "...", "dose": "...", "frequency": "...", "duration": 30, "category": "...", "type": "discharge", "notes": "...", "route": "oral"}]}'

        response = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text.strip()
        print("\n=== CLAUDE RAW RESPONSE ===")
        print(response_text)
        print("=== END ===\n")

        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()

        result = json.loads(response_text)
        drugs = result.get("drugs", [])

        formatted = []
        for d in drugs:
            drug_type = d.get("type", "outpatient")
            formatted.append({
                "drug_name":    d.get("name", "Unknown"),
                "generic":      d.get("name", "Unknown"),
                "category":     d.get("category", "Other"),
                "dosage":       d.get("dose", "See prescription"),
                "frequency":    d.get("frequency", "OD"),
                "freq_label":   freq_to_label(d.get("frequency", "OD")),
                "times":        freq_to_times(d.get("frequency", "OD")),
                "duration":     str(d.get("duration", 30)) + " days",
                "route":        d.get("route", "oral"),
                "type":         drug_type,
                "notes":        d.get("notes", ""),
                "confidence":   95,
                "needs_review": False,
                "ai_extracted": True,
                "active":       drug_type in ["discharge", "outpatient"],
            })
        print("=== EXTRACTED " + str(len(formatted)) + " DRUGS ===\n")
        return formatted

    except Exception as e:
        print("\n=== CLAUDE ERROR ===")
        print(str(e))
        print("=== END ===\n")
        return None


@app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "MedRecord AI",
        "version": "1.0.0"
    }


@app.post("/ocr/prescription")
async def ocr_prescription(file: UploadFile = File(...)):
    try:
        contents  = await file.read()
        image     = Image.open(io.BytesIO(contents))
        img_array = np.array(image)

        # Try Google Vision first (better for printed text)
        raw_text       = ""
        avg_confidence = 0

        try:
            img_bytes = io.BytesIO()
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(img_bytes, format='JPEG')
            b64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

            vision_resp = requests.post(
                "https://vision.googleapis.com/v1/images:annotate?key=" + GOOGLE_VISION_KEY,
                json={"requests": [{"image": {"content": b64}, "features": [{"type": "TEXT_DETECTION"}]}]},
                timeout=15
            )
            vision_data = vision_resp.json()
            raw_text = vision_data["responses"][0].get("fullTextAnnotation", {}).get("text", "")
            avg_confidence = 95 if raw_text else 0
            print("=== GOOGLE VISION OCR (length: " + str(len(raw_text)) + ") ===")
        except Exception as e:
            print("Google Vision failed: " + str(e))

        # Fall back to EasyOCR
        if not raw_text:
            enhanced = enhance_image(img_array)
            results  = reader.readtext(enhanced)
            confidence = 0
            for (bbox, text, conf) in results:
                raw_text   += text + "\n"
                confidence += conf
            avg_confidence = (confidence / len(results) * 100) if results else 0

        # Try Claude AI first
        drugs = extract_drugs_with_claude(raw_text)
        if drugs is None or len(drugs) == 0:
            drugs = parse_prescription(raw_text)

        return {
            "success":        True,
            "raw_text":       raw_text,
            "avg_confidence": round(avg_confidence, 1),
            "drugs_found":    len(drugs),
            "drugs":          drugs,
            "message":        "Found " + str(len(drugs)) + " medication(s)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/report")
async def ocr_report(file: UploadFile = File(...)):
    try:
        contents  = await file.read()
        image     = Image.open(io.BytesIO(contents))
        img_array = np.array(image)
        raw_text  = ""

        try:
            img_bytes = io.BytesIO()
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(img_bytes, format='JPEG')
            b64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            vision_resp = requests.post(
                "https://vision.googleapis.com/v1/images:annotate?key=" + GOOGLE_VISION_KEY,
                json={"requests": [{"image": {"content": b64}, "features": [{"type": "TEXT_DETECTION"}]}]},
                timeout=15
            )
            vision_data = vision_resp.json()
            raw_text = vision_data["responses"][0].get("fullTextAnnotation", {}).get("text", "")
        except Exception as e:
            print("Vision failed for report: " + str(e))

        if not raw_text:
            enhanced = enhance_image(img_array)
            results  = reader.readtext(enhanced)
            raw_text = "\n".join([text for (_, text, _) in results])

        return {
            "success":    True,
            "raw_text":   raw_text,
            "lab_values": [],
            "lab_name":   "Unknown Lab",
            "test_name":  "Lab Report",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)