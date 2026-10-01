import pandas as pd

DOMAIN_KEYWORDS = {
    "Customer": ["customer", "client", "user", "name", "phone", "email", "address", "age", "gender", "region", "zipcode"],
    "Sales": ["sale", "order", "quantity", "price", "amount", "revenue", "product", "discount", "transaction"],
    "Finance": ["salary", "balance", "credit", "debit", "tax", "profit", "loss", "invoice", "payment", "account", "bank"],
    "Healthcare": ["patient", "doctor", "hospital", "diagnosis", "blood", "pulse", "treatment", "medicine", "dosage", "medical"],
    "Education": ["student", "gpa", "grade", "marks", "course", "subject", "school", "university", "roll_no", "exam"]
}

def detect_domain(df: pd.DataFrame) -> dict:
    """
    Detects the likely dataset domain based on column name heuristics.
    Returns domain name, matched keywords, and explanation.
    """
    if df is None or df.empty:
        return {
            "domain": "General / Other",
            "confidence": "Low",
            "matched_keywords": [],
            "reasoning": "Dataset is empty or undefined."
        }

    cols = [str(c).lower().strip() for c in df.columns]
    
    scores = {domain: 0 for domain in DOMAIN_KEYWORDS}
    matched_map = {domain: [] for domain in DOMAIN_KEYWORDS}

    for col in cols:
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw in col:
                    scores[domain] += 1
                    matched_map[domain].append(f"column '{col}' matches keyword '{kw}'")

    best_domain = max(scores, key=scores.get)
    max_score = scores[best_domain]

    if max_score == 0:
        return {
            "domain": "General / Other",
            "confidence": "Default",
            "matched_keywords": [],
            "reasoning": "No domain-specific column keywords identified; defaulted to General / Other."
        }

    matches = matched_map[best_domain]
    reasoning = f"Identified as '{best_domain}' domain based on {len(matches)} matching column patterns: " + ", ".join(matches[:3])
    
    return {
        "domain": best_domain,
        "confidence": "High" if max_score >= 3 else "Moderate",
        "matched_keywords": matches,
        "reasoning": reasoning
    }
