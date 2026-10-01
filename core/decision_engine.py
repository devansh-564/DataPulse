"""
DataPulse Explainable Decision Layer
Traceable rule-based decision engine that details how quality issues are evaluated,
matched to policy rules, and authorized for execution.
"""

def generate_decision_trace(issues: list) -> list:
    """
    Transforms detected issue recommendations into an explainable step-by-step decision log.
    """
    decision_trace = []
    
    for idx, issue in enumerate(issues, start=1):
        target = issue["column"]
        itype = issue["issue_type"]
        mode = issue["mode"]
        action = issue["action"]
        count = issue["detected_count"]
        
        if mode == "AUTO":
            reasoning = f"Automated execution approved for {count} instance(s) of '{itype}' in column '{target}'. Action '{action}' is deterministic and non-destructive."
            status = "Approved for Execution (AUTO)"
        else:
            reasoning = f"Manual review flagged for {count} instance(s) of '{itype}' in column '{target}'. Action '{action}' presents data integrity risk and requires human sign-off."
            status = "Flagged for Manual Review"

        decision_trace.append({
            "Step": f"DEC-{idx:03d}",
            "Target Column": target,
            "Detected Issue": itype,
            "Affected Count": count,
            "Selected Action": action,
            "Decision Mode": mode,
            "Status": status,
            "Explainable Rationale": reasoning
        })

    return decision_trace
