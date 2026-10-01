import re
import pandas as pd
import numpy as np

def extract_entities_from_text(text: str):
    """
    Extracts key entity types (Emails, Phone numbers, Dates, Amounts, Names, IDs, Addresses)
    from unstructured document text using regular expressions and pattern recognition.
    """
    if not text:
        return {
            "emails": [], "phones": [], "dates": [], "amounts": [],
            "names": [], "ids": [], "organizations": [], "fields": {}
        }

    # 1. Emails
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    emails = list(set(re.findall(email_pattern, text)))

    # 2. Phone Numbers (Indian & International formats)
    phone_pattern = r'(?:\+?\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}|\b\d{10}\b'
    phones = list(set(re.findall(phone_pattern, text)))

    # 3. Dates (YYYY-MM-DD, DD/MM/YYYY, Month DD, YYYY)
    date_pattern = r'\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b'
    dates = list(set(re.findall(date_pattern, text, flags=re.IGNORECASE)))

    # 4. Currency Amounts / Financial Values (e.g. ₹4,500, $1200, 50000 INR)
    amount_pattern = r'(?:[₹$€£]\s?\d+(?:,\d+)*(?:\.\d+)?|\b\d+(?:,\d+)*(?:\.\d+)?\s?(?:INR|USD|EUR|GBP)\b)'
    amounts = list(set(re.findall(amount_pattern, text, flags=re.IGNORECASE)))

    # 5. Key-Value Field Pairs (e.g., "Customer Name: Rahul Sharma", "Order ID: ORD-992")
    kv_pattern = r'(?P<key>[A-Za-z\s_]{3,20}):\s*(?P<val>[^\n,;]{2,50})'
    kv_pairs = re.findall(kv_pattern, text)
    fields = {}
    names = []
    ids = []
    
    for key, val in kv_pairs:
        k_clean = key.strip().title()
        v_clean = val.strip()
        fields[k_clean] = v_clean
        k_lower = k_clean.lower()
        if any(term in k_lower for term in ['name', 'customer', 'client', 'employee']):
            names.append(v_clean)
        if any(term in k_lower for term in ['id', 'code', 'number', 'ssn', 'key']):
            ids.append(v_clean)

    # Heuristic Name Extraction fallback if key-value wasn't explicitly formatted
    if not names:
        name_line_match = re.findall(r'(?:Name|Customer|Client)\s*[:=]\s*([A-Za-z\s]+)', text, flags=re.IGNORECASE)
        if name_line_match:
            names = [n.strip() for n in name_line_match]

    return {
        "emails": emails,
        "phones": phones,
        "dates": dates,
        "amounts": amounts,
        "names": list(set(names)),
        "ids": list(set(ids)),
        "fields": fields
    }

def convert_unstructured_to_structured(text: str, document_metadata: dict = None) -> pd.DataFrame:
    """
    CONVERT UNSTRUCTURED -> STRUCTURED DATA:
    Parses unstructured text or document sections and converts detected entities & key-value
    records into a clean structured pandas DataFrame preview.
    """
    entities = extract_entities_from_text(text)
    
    records = []
    
    # Check if text contains structured records separated by blank lines or headers
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    if len(paragraphs) > 1 and any(":" in p for p in paragraphs):
        for idx, para in enumerate(paragraphs, start=1):
            p_ent = extract_entities_from_text(para)
            if p_ent["fields"] or p_ent["emails"] or p_ent["phones"] or p_ent["amounts"]:
                rec = {
                    "Record_ID": idx,
                    "Name": p_ent["names"][0] if p_ent["names"] else p_ent["fields"].get("Name", p_ent["fields"].get("Customer Name", "N/A")),
                    "Email": p_ent["emails"][0] if p_ent["emails"] else "N/A",
                    "Phone": p_ent["phones"][0] if p_ent["phones"] else "N/A",
                    "Date": p_ent["dates"][0] if p_ent["dates"] else "N/A",
                    "Amount": p_ent["amounts"][0] if p_ent["amounts"] else "N/A",
                    "ID_Code": p_ent["ids"][0] if p_ent["ids"] else "N/A",
                    "Extracted_Fields_Count": len(p_ent["fields"])
                }
                # Include additional custom fields
                for k, v in p_ent["fields"].items():
                    if k not in rec:
                        rec[k] = v
                records.append(rec)

    # Global Single/Fallback Summary Record if no distinct paragraph records found
    if not records:
        records.append({
            "Record_ID": 1,
            "Name": entities["names"][0] if entities["names"] else entities["fields"].get("Name", "N/A"),
            "Email": entities["emails"][0] if entities["emails"] else "N/A",
            "Phone": entities["phones"][0] if entities["phones"] else "N/A",
            "Date": entities["dates"][0] if entities["dates"] else "N/A",
            "Amount": entities["amounts"][0] if entities["amounts"] else "N/A",
            "ID_Code": entities["ids"][0] if entities["ids"] else "N/A",
            "Total_Fields_Detected": len(entities["fields"])
        })

    df_structured = pd.DataFrame(records)
    return df_structured

def assess_unstructured_document_quality(text: str, document_metadata: dict = None):
    """
    Computes Document Quality Score (0-100) and 6 unstructured document quality dimensions:
    1. Text Completeness
    2. Extraction Quality
    3. Field Completeness
    4. Format Consistency
    5. Entity Validity
    6. Metadata Completeness
    """
    if not text or len(text.strip()) == 0:
        return {
            "document_quality_score": 0.0,
            "quality_tier": "CRITICAL",
            "dimensions": {
                "Text Completeness": 0.0, "Extraction Quality": 0.0,
                "Field Completeness": 0.0, "Format Consistency": 0.0,
                "Entity Validity": 0.0, "Metadata Completeness": 0.0
            },
            "issues": [{"severity": "High", "issue": "Empty Document", "recommendation": "Check document source."}]
        }

    entities = extract_entities_from_text(text)
    metadata = document_metadata or {}

    # 1. Text Completeness (Length & non-empty character ratio)
    length = len(text)
    text_completeness = min(100.0, max(50.0, (length / 500.0) * 100.0))

    # 2. Extraction Quality (Ratio of clean alphabetic/numeric chars vs junk symbols)
    alphanumeric_count = sum(1 for c in text if c.isalnum() or c.isspace())
    extraction_quality = (alphanumeric_count / max(1, length)) * 100.0

    # 3. Field Completeness (Ratio of core fields detected: Name, Email, Phone, Date, Amount, ID)
    detected_types = 0
    if entities["names"] or "Name" in entities["fields"]: detected_types += 1
    if entities["emails"]: detected_types += 1
    if entities["phones"]: detected_types += 1
    if entities["dates"]: detected_types += 1
    if entities["amounts"]: detected_types += 1
    if entities["ids"]: detected_types += 1
    field_completeness = (detected_types / 6.0) * 100.0

    # 4. Entity Validity (Valid Email syntax check & phone length check)
    valid_entity_count = 0
    total_entity_count = len(entities["emails"]) + len(entities["phones"])
    
    for email in entities["emails"]:
        if re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', email):
            valid_entity_count += 1
    for phone in entities["phones"]:
        digits = re.sub(r'\D', '', phone)
        if 10 <= len(digits) <= 12:
            valid_entity_count += 1

    if total_entity_count == 0:
        entity_validity = 90.0
    else:
        entity_validity = (valid_entity_count / total_entity_count) * 100.0

    # 5. Format Consistency
    format_consistency = 95.0 if len(entities["fields"]) > 0 else 80.0

    # 6. Metadata Completeness
    meta_keys = ["num_pages", "length_chars", "length_words", "tables_found"]
    present_meta = sum(1 for k in meta_keys if k in metadata)
    metadata_completeness = (present_meta / len(meta_keys)) * 100.0 if meta_keys else 100.0

    # Overall Document Quality Score (Weighted average)
    weights = {
        "Text Completeness": 0.20,
        "Extraction Quality": 0.20,
        "Field Completeness": 0.25,
        "Format Consistency": 0.15,
        "Entity Validity": 0.10,
        "Metadata Completeness": 0.10
    }

    dimensions = {
        "Text Completeness": round(text_completeness, 2),
        "Extraction Quality": round(extraction_quality, 2),
        "Field Completeness": round(field_completeness, 2),
        "Format Consistency": round(format_consistency, 2),
        "Entity Validity": round(entity_validity, 2),
        "Metadata Completeness": round(metadata_completeness, 2)
    }

    doc_score = sum(dimensions[dim] * w for dim, w in weights.items())
    doc_score = round(doc_score, 2)

    if doc_score >= 90: tier = "EXCELLENT"
    elif doc_score >= 75: tier = "GOOD"
    elif doc_score >= 60: tier = "FAIR"
    else: tier = "NEEDS REVIEW"

    # Identify unstructured document quality issues
    doc_issues = []
    if field_completeness < 60.0:
        doc_issues.append({
            "severity": "Medium",
            "column": "Document Entities",
            "issue_type": "Low Field Completeness",
            "recommendation": "Key document fields (Email, Phone, Date) were not fully detected. Review formatting."
        })
    if entity_validity < 80.0:
        doc_issues.append({
            "severity": "High",
            "column": "Entity Syntax",
            "issue_type": "Invalid Entity Format",
            "recommendation": "Some extracted email or phone numbers contain syntax flaws. Flag for manual verification."
        })
    if extraction_quality < 85.0:
        doc_issues.append({
            "severity": "High",
            "column": "Text Extraction",
            "issue_type": "Corrupted Characters",
            "recommendation": "Document text contains noisy or special characters. Apply text normalization."
        })

    return {
        "document_quality_score": doc_score,
        "quality_tier": tier,
        "dimensions": dimensions,
        "weights": weights,
        "entities": entities,
        "issues": doc_issues
    }
