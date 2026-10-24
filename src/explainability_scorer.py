"""Explainability Scorer for both System A (Rule-Based) and System B (Retrieval-Grounded).
Evaluates:
1. Citation Presence (boolean)
2. Citation Correctness against gold reference policy passage / rule ID (boolean & overlap score)
"""

import re
from typing import Dict, Any, List

def normalize_text(text: str) -> str:
    """Normalizes text for keyword/ngram overlap scoring."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def compute_token_overlap(str1: str, str2: str) -> float:
    """Computes Jaccard token overlap between two strings."""
    tokens1 = set(normalize_text(str1).split())
    tokens2 = set(normalize_text(str2).split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


def score_system_a_explainability(
    output: Dict[str, Any],
    gold_scenario: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Scores explainability for System A (Rule-Based).
    Checks if rule_id and reason are present, and if the cited rule matches gold_rule_id.
    """
    rule_id = output.get("rule_id", "").strip()
    reason = output.get("reason", "").strip()
    gold_rule_id = gold_scenario.get("gold_rule_id", "").strip()
    gold_passage = gold_scenario.get("gold_policy_passage", "")

    citation_present = bool(rule_id and len(rule_id) > 2 and reason)

    # Clean rule IDs for comparison (e.g., 'HR-POL-001 §2.1' vs 'HR-POL-001 §2.1')
    norm_rule = normalize_text(rule_id)
    norm_gold_rule = normalize_text(gold_rule_id)

    # Check exact/partial rule match or token overlap with gold passage
    rule_match = (
        norm_rule in norm_gold_rule
        or norm_gold_rule in norm_rule
        or (rule_id.split("§")[0].strip() == gold_rule_id.split("§")[0].strip() and ("§" in rule_id and "§" in gold_rule_id and rule_id.split("§")[1][:3] == gold_rule_id.split("§")[1][:3]))
    )

    reason_overlap = compute_token_overlap(reason, gold_passage)
    citation_correct = citation_present and (rule_match or reason_overlap >= 0.15)

    return {
        "citation_present": citation_present,
        "citation_correct": bool(citation_correct),
        "rule_match": bool(rule_match),
        "reason_overlap_score": round(reason_overlap, 3),
        "cited_reference": rule_id,
        "gold_reference": gold_rule_id,
    }


def score_system_b_explainability(
    output: Dict[str, Any],
    gold_scenario: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Scores explainability for System B (Retrieval-Grounded).
    Checks if citation_text and citation_chunk_id are present, and if the cited passage aligns with gold policy passage.
    """
    citation_text = output.get("citation_text", "").strip()
    citation_chunk_id = output.get("citation_chunk_id", "").strip()
    reasoning = output.get("reasoning", "").strip()
    gold_passage = gold_scenario.get("gold_policy_passage", "")
    gold_chunk_id = gold_scenario.get("gold_chunk_id", "")

    citation_present = bool(citation_text and len(citation_text) > 10 and reasoning)

    # Check chunk match or text overlap with gold passage
    chunk_match = bool(gold_chunk_id and gold_chunk_id in citation_chunk_id)
    text_overlap = compute_token_overlap(citation_text + " " + reasoning, gold_passage)

    # Key terms match check
    gold_keywords = [w for w in normalize_text(gold_passage).split() if len(w) > 4]
    matched_keywords = [w for w in gold_keywords if w in normalize_text(citation_text + " " + reasoning)]
    keyword_coverage = (len(matched_keywords) / len(gold_keywords)) if gold_keywords else 0.0

    citation_correct = citation_present and (chunk_match or text_overlap >= 0.15 or keyword_coverage >= 0.35)

    return {
        "citation_present": citation_present,
        "citation_correct": bool(citation_correct),
        "chunk_match": bool(chunk_match),
        "text_overlap_score": round(text_overlap, 3),
        "keyword_coverage": round(keyword_coverage, 3),
        "cited_reference": citation_text[:120] if citation_text else citation_chunk_id,
        "gold_reference": gold_passage[:120],
    }
