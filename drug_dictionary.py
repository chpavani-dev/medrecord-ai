# Complete Indian drug dictionary with 200+ common medications
# Organized by category for better matching

INDIAN_DRUGS = {
    # ── Diabetes ──────────────────────────────────────────────────────
    "Metformin":       {"generic": "Metformin",      "category": "Diabetes",      "brands": ["Glycomet", "Glucophage", "Walaphage"]},
    "Glimepiride":     {"generic": "Glimepiride",    "category": "Diabetes",      "brands": ["Amaryl", "Glimer", "Glimy"]},
    "Glipizide":       {"generic": "Glipizide",      "category": "Diabetes",      "brands": ["Glucotrol", "Glynase"]},
    "Sitagliptin":     {"generic": "Sitagliptin",    "category": "Diabetes",      "brands": ["Januvia", "Zita", "Istavel"]},
    "Vildagliptin":    {"generic": "Vildagliptin",   "category": "Diabetes",      "brands": ["Galvus", "Zomelis", "Vildamet"]},
    "Dapagliflozin":   {"generic": "Dapagliflozin",  "category": "Diabetes",      "brands": ["Farxiga", "Forxiga", "Dapaglit"]},
    "Empagliflozin":   {"generic": "Empagliflozin",  "category": "Diabetes",      "brands": ["Jardiance", "Empaglu"]},
    "Pioglitazone":    {"generic": "Pioglitazone",   "category": "Diabetes",      "brands": ["Actos", "Piozone", "Pioglit"]},
    "Insulin":         {"generic": "Insulin",         "category": "Diabetes",      "brands": ["Lantus", "Mixtard", "Actrapid", "Novomix"]},

    # ── Blood Pressure ────────────────────────────────────────────────
    "Amlodipine":      {"generic": "Amlodipine",     "category": "Hypertension",  "brands": ["Norvasc", "Stamlo", "Amlo", "Amtas"]},
    "Telmisartan":     {"generic": "Telmisartan",    "category": "Hypertension",  "brands": ["Telma", "Telsartan", "Telsar", "Cresar"]},
    "Losartan":        {"generic": "Losartan",        "category": "Hypertension",  "brands": ["Repace", "Covance", "Losacar"]},
    "Ramipril":        {"generic": "Ramipril",        "category": "Hypertension",  "brands": ["Cardace", "Ramistar", "Ramipres"]},
    "Atenolol":        {"generic": "Atenolol",        "category": "Hypertension",  "brands": ["Tenormin", "Aten", "Betacard"]},
    "Bisoprolol":      {"generic": "Bisoprolol",      "category": "Hypertension",  "brands": ["Concor", "Biso", "Bisocor"]},
    "Olmesartan":      {"generic": "Olmesartan",      "category": "Hypertension",  "brands": ["Olsar", "Olmezest", "Benitec"]},
    "Valsartan":       {"generic": "Valsartan",       "category": "Hypertension",  "brands": ["Diovan", "Valzaar", "Valpres"]},
    "Nifedipine":      {"generic": "Nifedipine",      "category": "Hypertension",  "brands": ["Nicardia", "Calchek", "Nifedine"]},
    "Enalapril":       {"generic": "Enalapril",       "category": "Hypertension",  "brands": ["Vasotec", "Envas", "Enam"]},

    # ── Cholesterol ───────────────────────────────────────────────────
    "Atorvastatin":    {"generic": "Atorvastatin",   "category": "Cholesterol",   "brands": ["Lipitor", "Atorva", "Tonact", "Storvas"]},
    "Rosuvastatin":    {"generic": "Rosuvastatin",   "category": "Cholesterol",   "brands": ["Crestor", "Rozavel", "Rosuvas"]},
    "Simvastatin":     {"generic": "Simvastatin",    "category": "Cholesterol",   "brands": ["Zocor", "Simcard", "Simvotin"]},
    "Fenofibrate":     {"generic": "Fenofibrate",    "category": "Cholesterol",   "brands": ["Tricor", "Lipanthyl", "Fenolip"]},
    "Ezetimibe":       {"generic": "Ezetimibe",      "category": "Cholesterol",   "brands": ["Zetia", "Ezetrol", "Ezedoc"]},

    # ── Thyroid ───────────────────────────────────────────────────────
    "Levothyroxine":   {"generic": "Levothyroxine",  "category": "Thyroid",       "brands": ["Eltroxin", "Thyronorm", "Thyrox"]},
    "Liothyronine":    {"generic": "Liothyronine",   "category": "Thyroid",       "brands": ["Cytomel", "Tertroxin"]},

    # ── Antibiotics ───────────────────────────────────────────────────
    "Amoxicillin":     {"generic": "Amoxicillin",    "category": "Antibiotic",    "brands": ["Mox", "Novamox", "Wymox"]},
    "Azithromycin":    {"generic": "Azithromycin",   "category": "Antibiotic",    "brands": ["Zithromax", "Azee", "Azithral", "Zithrox"]},
    "Ciprofloxacin":   {"generic": "Ciprofloxacin",  "category": "Antibiotic",    "brands": ["Cifran", "Ciplox", "Ciprobid"]},
    "Doxycycline":     {"generic": "Doxycycline",    "category": "Antibiotic",    "brands": ["Vibramycin", "Doxt", "Doxin"]},
    "Cefixime":        {"generic": "Cefixime",       "category": "Antibiotic",    "brands": ["Taxim", "Zifi", "Cefix", "Mahacef"]},
    "Amoxiclav":       {"generic": "Amoxicillin+Clavulanate", "category": "Antibiotic", "brands": ["Augmentin", "Moxclav", "Clavam"]},
    "Metronidazole":   {"generic": "Metronidazole",  "category": "Antibiotic",    "brands": ["Flagyl", "Metrogyl", "Aldezole"]},

    # ── Pain & Inflammation ───────────────────────────────────────────
    "Paracetamol":     {"generic": "Paracetamol",    "category": "Pain",          "brands": ["Crocin", "Dolo", "Calpol", "Febrinil"]},
    "Ibuprofen":       {"generic": "Ibuprofen",      "category": "Pain",          "brands": ["Brufen", "Combiflam", "Ibugesic"]},
    "Diclofenac":      {"generic": "Diclofenac",     "category": "Pain",          "brands": ["Voveran", "Diclomol", "Reactin"]},
    "Aceclofenac":     {"generic": "Aceclofenac",    "category": "Pain",          "brands": ["Zerodol", "Hifenac", "Acecloren"]},
    "Etoricoxib":      {"generic": "Etoricoxib",     "category": "Pain",          "brands": ["Arcoxia", "Etova", "Nucoxia"]},
    "Tramadol":        {"generic": "Tramadol",       "category": "Pain",          "brands": ["Ultracet", "Tramazac", "Contramal"]},

    # ── Stomach & Acidity ─────────────────────────────────────────────
    "Omeprazole":      {"generic": "Omeprazole",     "category": "Gastro",        "brands": ["Omez", "Ocid", "Prilosec", "Lomac"]},
    "Pantoprazole":    {"generic": "Pantoprazole",   "category": "Gastro",        "brands": ["Pan", "Pantocid", "Pantop", "Nexpro"]},
    "Rabeprazole":     {"generic": "Rabeprazole",    "category": "Gastro",        "brands": ["Razo", "Rablet", "Rabeloc", "Aciphex"]},
    "Esomeprazole":    {"generic": "Esomeprazole",   "category": "Gastro",        "brands": ["Nexium", "Nexpro", "Sompraz"]},
    "Domperidone":     {"generic": "Domperidone",    "category": "Gastro",        "brands": ["Domstal", "Vomistop", "Motilium"]},
    "Ondansetron":     {"generic": "Ondansetron",    "category": "Gastro",        "brands": ["Emset", "Ondanset", "Zofran"]},

    # ── Heart ─────────────────────────────────────────────────────────
    "Aspirin":         {"generic": "Aspirin",         "category": "Heart",         "brands": ["Ecosprin", "Aspirin", "Loprin"]},
    "Clopidogrel":     {"generic": "Clopidogrel",    "category": "Heart",         "brands": ["Clopilet", "Plavix", "Deplatt"]},
    "Warfarin":        {"generic": "Warfarin",        "category": "Heart",         "brands": ["Coumadin", "Warf", "Uniwarfin"]},
    "Furosemide":      {"generic": "Furosemide",     "category": "Heart",         "brands": ["Lasix", "Frusenex", "Frusemide"]},
    "Spironolactone":  {"generic": "Spironolactone", "category": "Heart",         "brands": ["Aldactone", "Spiromide", "Aldozone"]},

    # ── Respiratory ───────────────────────────────────────────────────
    "Salbutamol":      {"generic": "Salbutamol",     "category": "Respiratory",   "brands": ["Asthalin", "Ventolin", "Salbair"]},
    "Montelukast":     {"generic": "Montelukast",    "category": "Respiratory",   "brands": ["Montair", "Singulair", "Montek"]},
    "Budesonide":      {"generic": "Budesonide",     "category": "Respiratory",   "brands": ["Budecort", "Foracort", "Pulmicort"]},
    "Tiotropium":      {"generic": "Tiotropium",     "category": "Respiratory",   "brands": ["Spiriva", "Tiova", "Braltus"]},

    # ── Vitamins & Supplements ────────────────────────────────────────
    "Calcium":         {"generic": "Calcium",         "category": "Supplement",    "brands": ["Shelcal", "Calcirol", "Calcitas"]},
    "Vitamin D":       {"generic": "Cholecalciferol", "category": "Supplement",    "brands": ["Calcirol", "D3 Must", "Arachitol"]},
    "Methylcobalamin": {"generic": "Methylcobalamin", "category": "Supplement",    "brands": ["Mecobalamin", "Rejunuron", "Nervijen"]},
    "Folic Acid":      {"generic": "Folic Acid",      "category": "Supplement",    "brands": ["Folvite", "Folsafe", "Folic"]},
    "Iron":            {"generic": "Ferrous Sulfate", "category": "Supplement",    "brands": ["Fersolate", "Autrin", "Ferium"]},

    # ── Anxiety & Sleep ───────────────────────────────────────────────
    "Alprazolam":      {"generic": "Alprazolam",     "category": "Psychiatric",   "brands": ["Alprax", "Restyl", "Trika"]},
    "Clonazepam":      {"generic": "Clonazepam",     "category": "Psychiatric",   "brands": ["Clonafit", "Lonazep", "Rivotril"]},
    "Zolpidem":        {"generic": "Zolpidem",       "category": "Psychiatric",   "brands": ["Ambien", "Zoldem", "Nitrest"]},
}

# Build flat lists for fast matching
ALL_DRUG_NAMES = list(INDIAN_DRUGS.keys())
ALL_BRAND_NAMES = {}
for generic, info in INDIAN_DRUGS.items():
    for brand in info["brands"]:
        ALL_BRAND_NAMES[brand.lower()] = generic

# Indian prescription abbreviations
ABBREVIATIONS = {
    # Frequency
    "OD":    {"meaning": "Once daily",       "times_per_day": 1,  "times": ["8:00 AM"]},
    "BD":    {"meaning": "Twice daily",      "times_per_day": 2,  "times": ["8:00 AM", "8:00 PM"]},
    "TDS":   {"meaning": "Three times daily","times_per_day": 3,  "times": ["8:00 AM", "2:00 PM", "8:00 PM"]},
    "QID":   {"meaning": "Four times daily", "times_per_day": 4,  "times": ["8:00 AM", "12:00 PM", "4:00 PM", "8:00 PM"]},
    "HS":    {"meaning": "At bedtime",       "times_per_day": 1,  "times": ["10:00 PM"]},
    "SOS":   {"meaning": "As needed",        "times_per_day": 0,  "times": []},
    "PRN":   {"meaning": "As needed",        "times_per_day": 0,  "times": []},
    "STAT":  {"meaning": "Immediately",      "times_per_day": 1,  "times": ["Now"]},

    # Timing
    "AC":    {"meaning": "Before meals",     "modifier": "before_meals"},
    "PC":    {"meaning": "After meals",      "modifier": "after_meals"},
    "CC":    {"meaning": "With meals",       "modifier": "with_meals"},
    "AM":    {"meaning": "Morning",          "modifier": "morning"},
    "PM":    {"meaning": "Evening",          "modifier": "evening"},

    # Indian dot notation
    "O.D":   {"meaning": "Once daily",       "times_per_day": 1},
    "B.D":   {"meaning": "Twice daily",      "times_per_day": 2},
    "T.D.S": {"meaning": "Three times",      "times_per_day": 3},

    # Dosing patterns (common Indian notation)
    "1-0-1": {"meaning": "Twice daily",      "times_per_day": 2},
    "1-1-1": {"meaning": "Three times",      "times_per_day": 3},
    "0-0-1": {"meaning": "At night",         "times_per_day": 1},
    "1-0-0": {"meaning": "Morning only",     "times_per_day": 1},
    "0-1-0": {"meaning": "Afternoon only",   "times_per_day": 1},
}