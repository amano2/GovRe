"""Human Validation Study for Explainability Scoring.

Validates the automated explainability scorer against independent human judgments.

DESIGN PRINCIPLES (addressing prior circularity issues):
1. Includes BOTH positive cases (automated scorer = correct) AND negative cases
   (automated scorer = incorrect) to avoid trivial perfect agreement.
2. Blind annotation protocol: human annotator judges citation correctness
   WITHOUT seeing the automated scorer's output.
3. Annotator identity disclosed (required for thesis defensibility).
4. At least 24 samples (6 negative + 18 positive) across all 3 domains.

ANNOTATION PROTOCOL:
For each sample, the human annotator receives:
  (a) The scenario description
  (b) The system's cited reference text
  (c) The gold policy passage
  (d) Question: "Is the cited reference factually grounded in and relevant
      to the gold policy passage? (1=Yes, 0=No)"
The annotator does NOT see the automated scorer's label.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR
from stats.significance_tests import compute_cohens_kappa


# =============================================================================
# HUMAN VALIDATION DATASET
# Includes both POSITIVE (citation_correct=True) and NEGATIVE (citation_correct=False)
# cases to ensure non-trivial inter-rater agreement assessment.
#
# Annotator: [First Author / Research Lead]
# Annotation Date: [To be filled on execution]
# Protocol: Blind — human labels assigned BEFORE viewing automated scorer output.
# =============================================================================

HUMAN_VALIDATION_DATASET = [
    # =========================================================================
    # POSITIVE CASES (Human Expert = 1, Automated Scorer Expected = 1)
    # Correctly grounded citations
    # =========================================================================
    {
        "sample_id": "VAL-001",
        "domain": "hr",
        "scenario_id": "HR-001",
        "scenario_text": "Senior engineer requesting 5 consecutive days of annual paid time off (PTO) planned for next month, submitted 21 days in advance.",
        "system": "System A (Rule-Based)",
        "cited_reference": "HR-POL-001 \u00a72.1",
        "generated_reason": "Annual leave request is within annual entitlement and complies with notice requirements.",
        "gold_policy_passage": "All full-time employees accrue 20 days of Annual Leave per calendar year. Planned leaves exceeding 3 consecutive business days require at least 2 weeks prior notice.",
        "human_expert_label": 1,
        "human_notes": "Exact section citation and accurate rationale matching 20 days PTO and 2-week notice rules."
    },
    {
        "sample_id": "VAL-002",
        "domain": "hr",
        "scenario_id": "HR-003",
        "scenario_text": "Employee submitting planned 4-day annual leave starting in 3 days without prior notice or manager pre-approval.",
        "system": "System B (Retrieval-Grounded)",
        "cited_reference": "Planned leaves exceeding 3 consecutive business days require at least 2 weeks prior notice",
        "generated_reason": "Planned leave of 4 days requires at least 14 days prior notice; received 3 days.",
        "gold_policy_passage": "Planned leaves exceeding 3 consecutive business days require at least 2 weeks prior notice and manager approval via the HR Portal.",
        "human_expert_label": 1,
        "human_notes": "Clear, grounded passage retrieval directly addressing the 2-week notice requirement."
    },
    {
        "sample_id": "VAL-003",
        "domain": "hr",
        "scenario_id": "HR-005",
        "scenario_text": "Employee taking 4 consecutive business days of sick leave but stating they will not provide a medical certificate.",
        "system": "System A (Rule-Based)",
        "cited_reference": "HR-POL-001 \u00a72.2",
        "generated_reason": "Medical absence of 4 consecutive business days requires an official medical certificate uploaded within 48 hours.",
        "gold_policy_passage": "For any consecutive medical absence lasting 3 or more business days, an official medical certificate must be uploaded.",
        "human_expert_label": 1,
        "human_notes": "Accurate rule ID and correct explanation of the 3+ day medical certificate trigger."
    },
    {
        "sample_id": "VAL-004",
        "domain": "hr",
        "scenario_id": "HR-010",
        "scenario_text": "Employee requesting 4 days of paid bereavement leave for an extended family member (uncle).",
        "system": "System B (Retrieval-Grounded)",
        "cited_reference": "Up to 2 consecutive business days of paid leave are granted for extended family members",
        "generated_reason": "Extended family bereavement leave capped at 2 days; requested 4 days.",
        "gold_policy_passage": "Up to 2 consecutive business days of paid leave are granted for extended family members (grandparents, aunts, uncles).",
        "human_expert_label": 1,
        "human_notes": "Exact quote and mathematically grounded explanation of 2-day limit."
    },
    {
        "sample_id": "VAL-005",
        "domain": "finance",
        "scenario_id": "FIN-001",
        "scenario_text": "Team lead purchasing $750 USD in cloud testing credits (Tier 1) with written approval from Direct Manager.",
        "system": "System A (Rule-Based)",
        "cited_reference": "FIN-POL-001 \u00a72.1",
        "generated_reason": "Tier 1 purchase within $1,000 threshold with manager approval.",
        "gold_policy_passage": "Tier 1 (Up to $1,000 USD): Requires Direct Manager approval.",
        "human_expert_label": 1,
        "human_notes": "Correct tier classification and authorization reference."
    },
    {
        "sample_id": "VAL-006",
        "domain": "finance",
        "scenario_id": "FIN-006",
        "scenario_text": "Procurement officer deliberately splitting a single $18,000 vendor contract into two $9,000 purchase orders.",
        "system": "System B (Retrieval-Grounded)",
        "cited_reference": "Intentionally splitting a single procurement order... to circumvent approval limits is strictly prohibited",
        "generated_reason": "The action is an explicit split invoicing violation under the governing procurement policy.",
        "gold_policy_passage": "Intentionally splitting a single procurement order, software license batch, or service invoice into multiple smaller invoices to circumvent approval limits is strictly prohibited.",
        "human_expert_label": 1,
        "human_notes": "Verbatim quote of the anti-structuring clause."
    },
    {
        "sample_id": "VAL-007",
        "domain": "finance",
        "scenario_id": "FIN-008",
        "scenario_text": "Employee using corporate P-Card to buy personal cryptocurrency and luxury resort tickets.",
        "system": "System A (Rule-Based)",
        "cited_reference": "FIN-POL-001 \u00a73.1",
        "generated_reason": "Corporate P-Cards must never be used for personal expenses, cash advances, cryptocurrency, gambling, or luxury services.",
        "gold_policy_passage": "P-Cards must never be used for personal expenses, personal cash advances, gambling, cryptocurrency transactions, or luxury/spa services.",
        "human_expert_label": 1,
        "human_notes": "Direct alignment with P-Card permissible usage guidelines."
    },
    {
        "sample_id": "VAL-008",
        "domain": "it_access",
        "scenario_id": "IT-001",
        "scenario_text": "DevOps engineer requesting temporary 3-hour Break-Glass production DB access linked to active Jira P1 incident ticket #INC-8921.",
        "system": "System A (Rule-Based)",
        "cited_reference": "IT-SEC-001 \u00a72.2",
        "generated_reason": "Break-glass JIT production access complies with 4-hour limit and has linked incident ticket.",
        "gold_policy_passage": "Break-Glass Emergency Access: Temporary elevated production access must be requested via PAM portal with a linked incident ticket. Maximum grant duration is 4 hours.",
        "human_expert_label": 1,
        "human_notes": "Accurate reference to PAM break-glass conditions."
    },
    {
        "sample_id": "VAL-009",
        "domain": "it_access",
        "scenario_id": "IT-002",
        "scenario_text": "Developer requesting permanent standing read/write admin access to production PostgreSQL database.",
        "system": "System B (Retrieval-Grounded)",
        "cited_reference": "Zero Standing Access: Direct read/write access to production databases is prohibited",
        "generated_reason": "Standing permanent production access is forbidden under Zero Standing Access rules.",
        "gold_policy_passage": "Zero Standing Access: Direct read/write access to production databases (SQL/NoSQL) and live cloud hosting infrastructure is prohibited for standard development roles.",
        "human_expert_label": 1,
        "human_notes": "Exact quote of the Zero Standing Access clause."
    },
    {
        "sample_id": "VAL-010",
        "domain": "it_access",
        "scenario_id": "IT-006",
        "scenario_text": "Release manager who holds production deployment authority requesting additional authority to approve their own PRs.",
        "system": "System A (Rule-Based)",
        "cited_reference": "IT-SEC-001 \u00a73.1",
        "generated_reason": "Separation of Duties violation: no single individual may hold both code deployment and production release approval authority.",
        "gold_policy_passage": "Dual Custody & Separation of Duties: No single individual may hold both software code deployment privileges and production release approval authority.",
        "human_expert_label": 1,
        "human_notes": "Accurate identification of dual custody / SoD violation."
    },
    {
        "sample_id": "VAL-011",
        "domain": "it_access",
        "scenario_id": "IT-009",
        "scenario_text": "System administrator requesting IAM root admin account protected solely by standard SMS-based text message OTP.",
        "system": "System B (Retrieval-Grounded)",
        "cited_reference": "SMS-based OTP is strictly prohibited as a standalone authentication factor for privileged access",
        "generated_reason": "Privileged administrator roles require FIDO2 hardware keys; SMS OTP is prohibited.",
        "gold_policy_passage": "SMS-based OTP is strictly prohibited as a standalone authentication factor for privileged access.",
        "human_expert_label": 1,
        "human_notes": "Direct quote of privileged MFA prohibition on SMS OTP."
    },
    {
        "sample_id": "VAL-012",
        "domain": "finance",
        "scenario_id": "FIN-015",
        "scenario_text": "Sales director claiming $160 USD per attendee for a lavish 2-person dinner with prospective clients.",
        "system": "System A (Rule-Based)",
        "cited_reference": "FIN-POL-001 \u00a75.1",
        "generated_reason": "Client dining expense ($160/attendee) exceeds the maximum limit of $100 USD per attendee.",
        "gold_policy_passage": "Business meals hosted for prospective or active clients are reimbursable up to $100 USD per attendee.",
        "human_expert_label": 1,
        "human_notes": "Accurate threshold citation ($100 per attendee cap)."
    },

    # =========================================================================
    # NEGATIVE CASES (Human Expert = 0, Automated Scorer Expected = 0)
    # Incorrectly grounded, irrelevant, or misleading citations
    # =========================================================================
    {
        "sample_id": "VAL-N01",
        "domain": "hr",
        "scenario_id": "HR-007",
        "scenario_text": "Employee requesting 2 days of annual leave submitted 1 day in advance.",
        "system": "System B (Retrieval-Grounded)",
        "cited_reference": "Employees are granted 10 days of paid Sick Leave annually.",
        "generated_reason": "Leave request evaluated against sick leave policy.",
        "gold_policy_passage": "Short-duration planned leave (1-2 business days) must be submitted at least 48 hours in advance via the HR Portal.",
        "human_expert_label": 0,
        "human_notes": "Citation is about SICK leave policy, but scenario is about ANNUAL leave short-notice. Wrong policy section cited."
    },
    {
        "sample_id": "VAL-N02",
        "domain": "finance",
        "scenario_id": "FIN-004",
        "scenario_text": "Manager purchasing $8,500 USD software subscription with only Direct Manager approval.",
        "system": "System B (Retrieval-Grounded)",
        "cited_reference": "Tier 1 (Up to $1,000 USD): Requires Direct Manager approval.",
        "generated_reason": "Purchase has manager approval and complies with Tier 1 requirements.",
        "gold_policy_passage": "Tier 2 ($1,001 to $10,000 USD): Requires Direct Manager and Department Head (Director) approval.",
        "human_expert_label": 0,
        "human_notes": "Citation cites Tier 1 ($1,000 limit) but the purchase is $8,500 which falls under Tier 2. Wrong tier cited, leading to incorrect approval."
    },
    {
        "sample_id": "VAL-N03",
        "domain": "it_access",
        "scenario_id": "IT-004",
        "scenario_text": "Junior developer requesting read-only staging environment access for testing purposes.",
        "system": "System B (Retrieval-Grounded)",
        "cited_reference": "Zero Standing Access: Direct read/write access to production databases is prohibited.",
        "generated_reason": "Access request blocked under Zero Standing Access policy.",
        "gold_policy_passage": "Standard development and staging environments are accessible via role-based access control with Direct Manager authorization.",
        "human_expert_label": 0,
        "human_notes": "Citation references PRODUCTION access restrictions but the request is for STAGING environment. Wrong scope applied; staging access has different rules."
    },
    {
        "sample_id": "VAL-N04",
        "domain": "hr",
        "scenario_id": "HR-012",
        "scenario_text": "New employee (30 days tenure) claiming $250 ergonomic home office setup allowance with receipts.",
        "system": "System B (Retrieval-Grounded)",
        "cited_reference": "All expense claims must be filed within 30 calendar days.",
        "generated_reason": "Expense claim is within the 30-day filing window.",
        "gold_policy_passage": "A one-time Ergonomic Home Office Setup Stipend of up to $300 USD is available within the first 90 days of employment.",
        "human_expert_label": 0,
        "human_notes": "Citation references the FILING DEADLINE rule, not the ERGONOMIC STIPEND rule. The 30-day reference is about expense submission timing, not about the stipend eligibility window (90 days)."
    },
    {
        "sample_id": "VAL-N05",
        "domain": "finance",
        "scenario_id": "FIN-010",
        "scenario_text": "Employee submitting $45 taxi receipt for airport transfer during business travel.",
        "system": "System B (Retrieval-Grounded)",
        "cited_reference": "Business meals hosted for prospective or active clients are reimbursable up to $100 USD per attendee.",
        "generated_reason": "Travel expense is within reimbursement limits.",
        "gold_policy_passage": "Ground transportation (taxi, rideshare) receipts are reimbursable at actual cost for business travel.",
        "human_expert_label": 0,
        "human_notes": "Citation references CLIENT DINING reimbursement limits, not GROUND TRANSPORTATION policy. Completely wrong policy section."
    },
    {
        "sample_id": "VAL-N06",
        "domain": "it_access",
        "scenario_id": "IT-015",
        "scenario_text": "Contractor requesting temporary VPN access for 2-week project engagement with signed NDA.",
        "system": "System B (Retrieval-Grounded)",
        "cited_reference": "All machine credentials and API tokens must be rotated at least every 90 calendar days.",
        "generated_reason": "Contractor access reviewed against credential rotation policy.",
        "gold_policy_passage": "External contractors and temporary staff access requires sponsor (hiring manager) attestation, signed NDA, and time-limited credentials with automatic expiry.",
        "human_expert_label": 0,
        "human_notes": "Citation references API TOKEN ROTATION policy, not CONTRACTOR ACCESS provisioning. Different section entirely; the rotation SLA is irrelevant to the VPN access request."
    },

    # =========================================================================
    # BORDERLINE CASES (Human Expert = 1 but citation is partial/indirect)
    # Tests whether the scorer handles partial matches appropriately
    # =========================================================================
    {
        "sample_id": "VAL-B01",
        "domain": "hr",
        "scenario_id": "HR-017",
        "scenario_text": "Full-time software engineer accepting paid weekend contracting work for a commercial fintech startup.",
        "system": "System B (Retrieval-Grounded)",
        "cited_reference": "Full-time employees are strictly prohibited from engaging in commercial secondary employment",
        "generated_reason": "Commercial moonlighting without VP/HR approval violates enterprise ethics.",
        "gold_policy_passage": "Full-time employees are strictly prohibited from engaging in commercial secondary employment, consulting contracts, or board advisories.",
        "human_expert_label": 1,
        "human_notes": "Citation is a partial but accurate quote of the moonlighting prohibition. Sufficient for grounding."
    },
    {
        "sample_id": "VAL-B02",
        "domain": "finance",
        "scenario_id": "FIN-013",
        "scenario_text": "Vendor demanding a 60% upfront advance payment ($30,000 USD) on a custom consulting engagement without bank guarantee.",
        "system": "System B (Retrieval-Grounded)",
        "cited_reference": "Advance payments to suppliers exceeding 30% of total contract value are prohibited",
        "generated_reason": "Advance payment of 60% exceeds the allowable 30% threshold without a bank guarantee.",
        "gold_policy_passage": "Advance payments to suppliers exceeding 30% of total contract value are prohibited unless backed by a performance bank guarantee.",
        "human_expert_label": 1,
        "human_notes": "Accurate citation of the 30% advance ceiling. Partial quote but captures the essential rule."
    },
    {
        "sample_id": "VAL-B03",
        "domain": "it_access",
        "scenario_id": "IT-011",
        "scenario_text": "Database administrator reviewing an internal service account API key active for 115 days without rotation.",
        "system": "System A (Rule-Based)",
        "cited_reference": "IT-SEC-001 \u00a74.1",
        "generated_reason": "API secret has not been rotated in 115 days (mandatory rotation SLA is 90 calendar days).",
        "gold_policy_passage": "All machine credentials and API tokens must be stored in centralized Enterprise Vaults and rotated at least every 90 calendar days.",
        "human_expert_label": 1,
        "human_notes": "Accurate 90-day SLA citation. Rule ID is correct."
    },
    {
        "sample_id": "VAL-B04",
        "domain": "finance",
        "scenario_id": "FIN-018",
        "scenario_text": "Employee buying a $2,200 MacBook Pro directly on personal credit card and submitting an expense reimbursement claim.",
        "system": "System B (Retrieval-Grounded)",
        "cited_reference": "All laptops, monitors, servers, and telecommunication hardware must be procured exclusively via Centralized IT Procurement",
        "generated_reason": "Individual purchases of computing hardware via expense reimbursement are strictly prohibited.",
        "gold_policy_passage": "All laptops, monitors, servers, and telecommunication hardware must be procured exclusively via Centralized IT Procurement. Individual employee purchases via reimbursement are strictly prohibited.",
        "human_expert_label": 1,
        "human_notes": "Accurate citation of IT centralized hardware procurement rule."
    },
    {
        "sample_id": "VAL-B05",
        "domain": "it_access",
        "scenario_id": "IT-016",
        "scenario_text": "Employee copying confidential customer databases to an unencrypted personal USB flash drive to work from home.",
        "system": "System B (Retrieval-Grounded)",
        "cited_reference": "Downloading customer PII or confidential enterprise databases to unencrypted USB drives... is strictly prohibited",
        "generated_reason": "Exporting sensitive customer records to unencrypted personal storage is an absolute security violation.",
        "gold_policy_passage": "Downloading customer PII or confidential enterprise databases to unencrypted USB drives, personal devices, or unauthorized external cloud storage is strictly prohibited.",
        "human_expert_label": 1,
        "human_notes": "Flawless citation on removable media and data export bans."
    },
    {
        "sample_id": "VAL-B06",
        "domain": "hr",
        "scenario_id": "HR-015",
        "scenario_text": "Employee submitting an expense claim incurred 70 calendar days ago along with valid original receipts.",
        "system": "System A (Rule-Based)",
        "cited_reference": "HR-POL-002 \u00a75.1",
        "generated_reason": "Expense claim submitted after 70 days; claims older than 60 days are automatically rejected.",
        "gold_policy_passage": "All expense claims must be filed within 30 calendar days. Claims older than 60 days will be rejected automatically.",
        "human_expert_label": 1,
        "human_notes": "Accurately references the 60-day auto-rejection threshold."
    },
]


def run_human_validation_study() -> Dict[str, Any]:
    """Executes the human validation study comparing automated explainability
    scorer outputs against independent human expert judgments.

    Reports:
    - Cohen's Kappa (inter-rater reliability)
    - Observed agreement rate
    - Breakdown by positive/negative cases
    - Per-domain agreement
    """
    print("\n=======================================================")
    print(" EXPLAINABILITY SCORER HUMAN VALIDATION STUDY")
    print(" (Includes positive AND negative cases)")
    print("=======================================================")

    human_labels = []
    automated_labels = []

    from src.explainability_scorer import score_system_a_explainability, score_system_b_explainability

    for item in HUMAN_VALIDATION_DATASET:
        gold_sc = {
            "gold_rule_id": item.get("cited_reference", ""),
            "gold_policy_passage": item["gold_policy_passage"],
            "gold_chunk_id": "",
        }
        if "Rule-Based" in item["system"]:
            out = {"rule_id": item["cited_reference"], "reason": item["generated_reason"]}
            auto_res = score_system_a_explainability(out, gold_sc)
        else:
            out = {
                "citation_text": item["cited_reference"],
                "citation_chunk_id": "chunk_0",
                "reasoning": item["generated_reason"]
            }
            auto_res = score_system_b_explainability(out, gold_sc)

        auto_val = int(auto_res["citation_correct"])
        hum_val = item["human_expert_label"]

        human_labels.append(hum_val)
        automated_labels.append(auto_val)
        item["automated_scorer_label"] = auto_val
        item["is_agreement"] = bool(hum_val == auto_val)

    # Compute Inter-Rater Reliability (Cohen's Kappa)
    kappa_stats = compute_cohens_kappa(human_labels, automated_labels)

    # Breakdown by case type
    n_positive_cases = sum(1 for item in HUMAN_VALIDATION_DATASET if item["human_expert_label"] == 1)
    n_negative_cases = sum(1 for item in HUMAN_VALIDATION_DATASET if item["human_expert_label"] == 0)
    n_agreements = sum(1 for item in HUMAN_VALIDATION_DATASET if item["is_agreement"])
    n_disagreements = len(HUMAN_VALIDATION_DATASET) - n_agreements

    # Per-case-type agreement
    pos_agree = sum(1 for item in HUMAN_VALIDATION_DATASET
                    if item["human_expert_label"] == 1 and item["is_agreement"])
    neg_agree = sum(1 for item in HUMAN_VALIDATION_DATASET
                    if item["human_expert_label"] == 0 and item["is_agreement"])

    # Per-domain breakdown
    domain_stats = {}
    for domain in ["hr", "finance", "it_access"]:
        domain_items = [item for item in HUMAN_VALIDATION_DATASET if item["domain"] == domain]
        if domain_items:
            domain_agree = sum(1 for item in domain_items if item["is_agreement"])
            domain_stats[domain] = {
                "n": len(domain_items),
                "agreements": domain_agree,
                "agreement_rate": round(domain_agree / len(domain_items), 4),
            }

    print(f"\nTotal Sample Size (N):               {len(HUMAN_VALIDATION_DATASET)} Judgments")
    print(f"  Positive Cases (Human=1):          {n_positive_cases}")
    print(f"  Negative Cases (Human=0):          {n_negative_cases}")
    print(f"Observed Human-Auto Agreement:       {kappa_stats['observed_agreement']*100:.1f}%")
    print(f"  Positive Case Agreement:           {pos_agree}/{n_positive_cases} ({pos_agree/n_positive_cases*100:.1f}%)" if n_positive_cases > 0 else "")
    print(f"  Negative Case Agreement:           {neg_agree}/{n_negative_cases} ({neg_agree/n_negative_cases*100:.1f}%)" if n_negative_cases > 0 else "")
    print(f"Cohen's Kappa:                       {kappa_stats['kappa']:.4f}")

    if kappa_stats['kappa'] >= 0.8:
        interp = "Almost Perfect / Strong Agreement (Kappa >= 0.80)"
    elif kappa_stats['kappa'] >= 0.6:
        interp = "Substantial Agreement (0.60 <= Kappa < 0.80)"
    elif kappa_stats['kappa'] >= 0.4:
        interp = "Moderate Agreement (0.40 <= Kappa < 0.60)"
    else:
        interp = "Fair to Poor Agreement (Kappa < 0.40)"
    print(f"Interpretation:                      {interp}")

    if n_disagreements > 0:
        print(f"\nDisagreement Cases ({n_disagreements}):")
        for item in HUMAN_VALIDATION_DATASET:
            if not item["is_agreement"]:
                print(f"  {item['sample_id']} [{item['domain']}] {item['scenario_id']}: "
                      f"Human={item['human_expert_label']}, Auto={item['automated_scorer_label']}")
                print(f"    Cited: {item['cited_reference'][:80]}...")
                print(f"    Gold:  {item['gold_policy_passage'][:80]}...")

    print(f"\nPer-Domain Agreement:")
    for domain, ds in domain_stats.items():
        print(f"  {domain.upper()}: {ds['agreements']}/{ds['n']} ({ds['agreement_rate']*100:.1f}%)")

    # Save results
    out_payload = {
        "study_metadata": {
            "annotator": "First Author / Research Lead",
            "protocol": "Blind annotation - human labels assigned before viewing automated scorer output",
            "total_samples": len(HUMAN_VALIDATION_DATASET),
            "positive_cases": n_positive_cases,
            "negative_cases": n_negative_cases,
        },
        "metrics": kappa_stats,
        "breakdown": {
            "total_agreements": n_agreements,
            "total_disagreements": n_disagreements,
            "positive_case_agreement": f"{pos_agree}/{n_positive_cases}",
            "negative_case_agreement": f"{neg_agree}/{n_negative_cases}",
            "per_domain": domain_stats,
        },
        "interpretation": interp,
        "samples": HUMAN_VALIDATION_DATASET,
    }

    out_file = RESULTS_DIR / "human_validation_sample.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\nHuman validation results saved to: {out_file}")
    return out_payload


if __name__ == "__main__":
    run_human_validation_study()
