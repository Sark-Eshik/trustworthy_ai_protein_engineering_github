# src/reliability/reliability_classifier.py
"""Category definition and mapping helper for Reliability Scores.

Extends reliability classification logic to external modules.
"""

def map_score_to_category(score: float) -> str:
    """Classifies standard engineering category from a raw reliability score.
    
    Categories:
    - High Reliability: [0.75, 1.00]
    - Moderate Reliability: [0.50, 0.75)
    - Low Reliability: [0.25, 0.50)
    - Very Low Reliability: [0.00, 0.25)
    """
    if score >= 0.75:
        return "High"
    elif score >= 0.50:
        return "Moderate"
    elif score >= 0.25:
        return "Low"
    else:
        return "Very Low"
