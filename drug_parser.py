import re
from fuzzywuzzy import process, fuzz
from drug_dictionary import INDIAN_DRUGS, ALL_DRUG_NAMES, ALL_BRAND_NAMES, ABBREVIATIONS

def parse_prescription(raw_text):
    """
    Parse raw OCR text and extract structured drug data.
    Returns list of drugs with confidence scores.
    """
    lines  = raw_text.split('\n')
    lines  = [l.strip() for l in lines if l.strip()]
    drugs  = []
    seen   = set()

    for i, line in enumerate(lines):
        # Try to find a drug in this line
        drug_match = find_drug(line)

        if drug_match:
            drug_name   = drug_match["name"]
            generic     = drug_match["generic"]
            confidence  = drug_match["confidence"]

            # Skip duplicates
            if generic.lower() in seen:
                continue
            seen.add(generic.lower())

            # Look for dosage in same line or next 2 lines
            search_text = " ".join(lines[i:i+3])
            dosage      = extract_dosage(search_text)
            frequency   = extract_frequency(search_text)
            duration    = extract_duration(search_text)
            route       = extract_route(search_text)

            drugs.append({
                "drug_name":    drug_name,
                "generic":      generic,
                "category":     INDIAN_DRUGS.get(generic, {}).get("category", "Unknown"),
                "dosage":       dosage,
                "frequency":    frequency["code"],
                "freq_label":   frequency["label"],
                "times":        frequency["times"],
                "duration":     duration,
                "route":        route,
                "confidence":   confidence,
                "needs_review": confidence < 75
            })

    return drugs


def find_drug(text):
    """
    Find drug name in a line of text.
    Uses exact match first, then fuzzy matching.
    """
    text_upper = text.upper()
    text_lower = text.lower()

    # 1. Exact match against generic names
    for generic in ALL_DRUG_NAMES:
        if generic.upper() in text_upper:
            return {
                "name":       generic,
                "generic":    generic,
                "confidence": 100
            }

    # 2. Exact match against brand names
    for brand, generic in ALL_BRAND_NAMES.items():
        if brand in text_lower:
            return {
                "name":       brand.title(),
                "generic":    generic,
                "confidence": 95
            }

    # 3. Fuzzy match — catches spelling errors in handwriting
    # Only try if line has a capital letter (likely a drug name)
    words = [w for w in text.split() if len(w) > 3]
    for word in words:
        # Try against generic names
        result = process.extractOne(
            word, ALL_DRUG_NAMES,
            scorer=fuzz.ratio
        )
        if result and result[1] >= 75:
            return {
                "name":       word,
                "generic":    result[0],
                "confidence": result[1]
            }

        # Try against brand names
        brand_result = process.extractOne(
            word.lower(), list(ALL_BRAND_NAMES.keys()),
            scorer=fuzz.ratio
        )
        if brand_result and brand_result[1] >= 75:
            generic = ALL_BRAND_NAMES[brand_result[0]]
            return {
                "name":       word,
                "generic":    generic,
                "confidence": brand_result[1]
            }

    return None


def extract_dosage(text):
    """Extract dosage — e.g. 500mg, 10mg, 2.5mg"""
    patterns = [
        r'(\d+\.?\d*)\s*(mg|mcg|ml|g|IU|units?)',
        r'(\d+\.?\d*)\s*(milligrams?|micrograms?|millilitres?)',
        r'TAB\s+(\d+\.?\d*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            unit  = match.group(2) if match.lastindex >= 2 else "mg"
            return f"{value}{unit}"
    return "See prescription"


def extract_frequency(text):
    """Extract frequency with Indian abbreviation support"""
    text_upper = text.upper()

    # Check dot notation first (B.D, T.D.S)
    dot_map = {
        "T.D.S": "TDS", "B.D": "BD", "O.D": "OD",
        "Q.I.D": "QID", "H.S": "HS", "S.O.S": "SOS"
    }
    for dot, code in dot_map.items():
        if dot in text_upper:
            return build_freq(code)

    # Check dosing patterns (1-0-1, 1-1-1)
    pattern_map = {
        "1-1-1-1": "QID", "1-1-1": "TDS",
        "1-0-1":   "BD",  "0-0-1": "HS",
        "1-0-0":   "OD",  "0-1-0": "OD"
    }
    for pattern, code in pattern_map.items():
        if pattern in text:
            return build_freq(code)

    # Check standard abbreviations
    freq_priority = ["QID", "TDS", "BD", "HS", "SOS", "PRN", "STAT", "OD"]
    for code in freq_priority:
        if re.search(r'\b' + code + r'\b', text_upper):
            return build_freq(code)

    # Check natural language
    natural_map = {
        "four times":  "QID",
        "three times": "TDS",
        "twice":       "BD",
        "twice daily": "BD",
        "once daily":  "OD",
        "at night":    "HS",
        "bedtime":     "HS",
        "as needed":   "SOS",
    }
    text_lower = text.lower()
    for phrase, code in natural_map.items():
        if phrase in text_lower:
            return build_freq(code)

    # Default to once daily
    return build_freq("OD")


def build_freq(code):
    """Build frequency object from abbreviation code"""
    abbr = ABBREVIATIONS.get(code, {})
    return {
        "code":  code,
        "label": abbr.get("meaning", code),
        "times": abbr.get("times", ["8:00 AM"]),
    }


def extract_duration(text):
    """Extract treatment duration — e.g. 5 days, 2 weeks"""
    patterns = [
        r'(\d+)\s*(days?|d)',
        r'(\d+)\s*(weeks?|wks?|w)',
        r'(\d+)\s*(months?|mths?|m)',
        r'for\s+(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num  = match.group(1)
            unit = match.group(2) if match.lastindex >= 2 else "days"
            unit = unit.lower().rstrip('s')
            if 'week' in unit or unit == 'w':
                return f"{int(num) * 7} days"
            if 'month' in unit or unit == 'm':
                return f"{int(num) * 30} days"
            return f"{num} days"
    return "30 days"


def extract_route(text):
    """Extract route of administration"""
    routes = {
        "oral":       ["oral", "po", "by mouth", "tablet", "tab", "cap", "capsule", "syrup"],
        "topical":    ["topical", "apply", "cream", "ointment", "gel", "lotion"],
        "injection":  ["injection", "im", "iv", "sc", "inj", "subcutaneous", "intramuscular"],
        "inhaler":    ["inhale", "inhaler", "puff", "mdi", "nebulize"],
        "eye drops":  ["eye", "ophthalmic", "drops", "instill"],
        "ear drops":  ["ear", "otic"],
        "nasal":      ["nasal", "nose", "nostril"],
    }
    text_lower = text.lower()
    for route, keywords in routes.items():
        if any(k in text_lower for k in keywords):
            return route
    return "oral"