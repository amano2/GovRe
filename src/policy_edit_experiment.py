"""Policy-Change Robustness Experiment (Headline Result).

Evaluates whether Retrieval-Grounded Governance (System B) adapts correctly
to edited policy documents via re-indexing alone without code modifications,
versus Rule-Based Governance (System A) which goes stale.

Tracks response_source for every System B evaluation to ensure results
are not contaminated by API failures.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DOMAINS, RESULTS_DIR, EVAL_SEEDS
from src.index import DomainIndex, build_all_indexes
from src.rule_engine import evaluate_hr_rule, evaluate_finance_rule, evaluate_it_access_rule
from src.retrieval_governance import RetrievalGovernanceJudge
from src.explainability_scorer import score_system_a_explainability, score_system_b_explainability
from stats.significance_tests import compute_bootstrap_ci, mcnemar_test

RULE_EVALUATORS = {
    "hr": evaluate_hr_rule,
    "finance": evaluate_finance_rule,
    "it_access": evaluate_it_access_rule,
}

# Post-Policy-Edit Benchmark Scenarios (Specifically probing the edited policy rules, N=30)
POST_EDIT_SCENARIOS = {
    "hr": [
        {
            "id": "HR-EDIT-001",
            "domain": "hr",
            "category": "leave",
            "text": "Employee requesting 4 consecutive days of planned annual leave submitted 7 business days in advance.",
            "details": {"leave_type": "annual_pto", "days_requested": 4, "notice_days": 7},
            "original_policy_gold": "FLAG",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "HR-POL-001 §2.1",
            "gold_chunk_id": "hr_leave_policy_chunk_2",
            "gold_policy_passage": "Planned leaves exceeding 3 consecutive business days require at least 5 business days prior notice and manager approval."
        },
        {
            "id": "HR-EDIT-002",
            "domain": "hr",
            "category": "leave",
            "text": "Employee submitting 4 consecutive days of annual leave with only 2 days notice.",
            "details": {"leave_type": "annual_pto", "days_requested": 4, "notice_days": 2},
            "original_policy_gold": "FLAG",
            "edited_policy_gold": "FLAG",
            "gold_decision": "FLAG",
            "gold_rule_id": "HR-POL-001 §2.1",
            "gold_chunk_id": "hr_leave_policy_chunk_2",
            "gold_policy_passage": "Planned leaves exceeding 3 consecutive business days require at least 5 business days prior notice."
        },
        {
            "id": "HR-EDIT-003",
            "domain": "hr",
            "category": "leave",
            "text": "Employee applying for 15 days of annual PTO with 3 weeks notice.",
            "details": {"leave_type": "annual_pto", "days_requested": 15, "notice_days": 21},
            "original_policy_gold": "APPROVE",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "HR-POL-001 §2.1",
            "gold_chunk_id": "hr_leave_policy_chunk_2",
            "gold_policy_passage": "All full-time employees accrue 20 days of Annual Leave per calendar year."
        },
        {
            "id": "HR-EDIT-004",
            "domain": "hr",
            "category": "leave",
            "text": "Employee requesting 5 days annual leave submitted 6 business days in advance.",
            "details": {"leave_type": "annual_pto", "days_requested": 5, "notice_days": 6},
            "original_policy_gold": "FLAG",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "HR-POL-001 §2.1",
            "gold_chunk_id": "hr_leave_policy_chunk_2",
            "gold_policy_passage": "Planned leaves exceeding 3 consecutive business days require at least 5 business days prior notice."
        },
        {
            "id": "HR-EDIT-005",
            "domain": "hr",
            "category": "leave",
            "text": "Employee submitting 4 days annual leave exactly 5 business days in advance.",
            "details": {"leave_type": "annual_pto", "days_requested": 4, "notice_days": 5},
            "original_policy_gold": "FLAG",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "HR-POL-001 §2.1",
            "gold_chunk_id": "hr_leave_policy_chunk_2",
            "gold_policy_passage": "Planned leaves exceeding 3 consecutive business days require at least 5 business days prior notice."
        },
        {
            "id": "HR-EDIT-006",
            "domain": "hr",
            "category": "leave",
            "text": "Employee submitting 4 days annual leave with only 4 business days notice.",
            "details": {"leave_type": "annual_pto", "days_requested": 4, "notice_days": 4},
            "original_policy_gold": "FLAG",
            "edited_policy_gold": "FLAG",
            "gold_decision": "FLAG",
            "gold_rule_id": "HR-POL-001 §2.1",
            "gold_chunk_id": "hr_leave_policy_chunk_2",
            "gold_policy_passage": "Planned leaves exceeding 3 consecutive business days require at least 5 business days prior notice."
        },
        {
            "id": "HR-EDIT-007",
            "domain": "hr",
            "category": "leave",
            "text": "Employee taking 2 days casual leave submitted 48 hours in advance.",
            "details": {"leave_type": "annual_pto", "days_requested": 2, "notice_hours": 48},
            "original_policy_gold": "APPROVE",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "HR-POL-001 §3.2",
            "gold_chunk_id": "hr_leave_policy_chunk_3",
            "gold_policy_passage": "Requests for casual/planned leaves of 1–2 days must be submitted at least 48 hours in advance."
        },
        {
            "id": "HR-EDIT-008",
            "domain": "hr",
            "category": "leave",
            "text": "Senior engineer requesting 6 days annual PTO submitted 8 business days in advance.",
            "details": {"leave_type": "annual_pto", "days_requested": 6, "notice_days": 8},
            "original_policy_gold": "FLAG",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "HR-POL-001 §2.1",
            "gold_chunk_id": "hr_leave_policy_chunk_2",
            "gold_policy_passage": "Planned leaves exceeding 3 consecutive business days require at least 5 business days prior notice."
        },
        {
            "id": "HR-EDIT-009",
            "domain": "hr",
            "category": "leave",
            "text": "Lead researcher requesting 10 days of annual leave with 11 business days notice.",
            "details": {"leave_type": "annual_pto", "days_requested": 10, "notice_days": 11},
            "original_policy_gold": "FLAG",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "HR-POL-001 §2.1",
            "gold_chunk_id": "hr_leave_policy_chunk_2",
            "gold_policy_passage": "Planned leaves exceeding 3 consecutive business days require at least 5 business days prior notice."
        },
        {
            "id": "HR-EDIT-010",
            "domain": "hr",
            "category": "leave",
            "text": "Employee requesting 25 consecutive days of annual leave in a single application exceeding yearly accrual limit.",
            "details": {"leave_type": "annual_pto", "days_requested": 25, "notice_days": 30},
            "original_policy_gold": "BLOCK",
            "edited_policy_gold": "BLOCK",
            "gold_decision": "BLOCK",
            "gold_rule_id": "HR-POL-001 §2.1",
            "gold_chunk_id": "hr_leave_policy_chunk_2",
            "gold_policy_passage": "All full-time employees accrue 20 days of Annual Leave per calendar year."
        }
    ],
    "finance": [
        {
            "id": "FIN-EDIT-001",
            "domain": "finance",
            "category": "pcard",
            "text": "Employee submitting P-Card charge of $38 USD without itemized receipt on day 4.",
            "details": {"expense_type": "field_supplies", "amount_usd": 38.0, "days_since_transaction": 4, "has_itemized_receipt": False},
            "original_policy_gold": "FLAG",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "FIN-POL-001 §3.2",
            "gold_chunk_id": "finance_expense_and_procurement_policy_chunk_3",
            "gold_policy_passage": "Transactions exceeding $50 USD require an itemized digital tax invoice."
        },
        {
            "id": "FIN-EDIT-002",
            "domain": "finance",
            "category": "procurement",
            "text": "Department purchasing $13,500 USD software package with Direct Manager and Department Director sign-offs.",
            "details": {"contract_amount_usd": 13500.0, "approvals_obtained": ["manager", "director"], "vendor_quotes_count": 1},
            "original_policy_gold": "FLAG",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "FIN-POL-001 §2.1",
            "gold_chunk_id": "finance_expense_and_procurement_policy_chunk_2",
            "gold_policy_passage": "Tier 2 ($1,001 to $15,000 USD): Requires Direct Manager and Department Head (Director) approval."
        },
        {
            "id": "FIN-EDIT-003",
            "domain": "finance",
            "category": "pcard",
            "text": "Employee submitting P-Card charge of $65 USD without itemized receipt on day 4.",
            "details": {"expense_type": "field_supplies", "amount_usd": 65.0, "days_since_transaction": 4, "has_itemized_receipt": False},
            "original_policy_gold": "FLAG",
            "edited_policy_gold": "FLAG",
            "gold_decision": "FLAG",
            "gold_rule_id": "FIN-POL-001 §3.2",
            "gold_chunk_id": "finance_expense_and_procurement_policy_chunk_3",
            "gold_policy_passage": "Transactions exceeding $50 USD require an itemized digital tax invoice."
        },
        {
            "id": "FIN-EDIT-004",
            "domain": "finance",
            "category": "pcard",
            "text": "Employee submitting P-Card expense of $45 USD for office supplies without receipt on day 3.",
            "details": {"expense_type": "office_supplies", "amount_usd": 45.0, "days_since_transaction": 3, "has_itemized_receipt": False},
            "original_policy_gold": "FLAG",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "FIN-POL-001 §3.2",
            "gold_chunk_id": "finance_expense_and_procurement_policy_chunk_3",
            "gold_policy_passage": "Transactions exceeding $50 USD require an itemized digital tax invoice."
        },
        {
            "id": "FIN-EDIT-005",
            "domain": "finance",
            "category": "pcard",
            "text": "Employee expensing $49.50 USD team lunch on P-Card without invoice on day 5.",
            "details": {"expense_type": "team_lunch", "amount_usd": 49.5, "days_since_transaction": 5, "has_itemized_receipt": False},
            "original_policy_gold": "FLAG",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "FIN-POL-001 §3.2",
            "gold_chunk_id": "finance_expense_and_procurement_policy_chunk_3",
            "gold_policy_passage": "Transactions exceeding $50 USD require an itemized digital tax invoice."
        },
        {
            "id": "FIN-EDIT-006",
            "domain": "finance",
            "category": "procurement",
            "text": "Procurement of $11,000 USD cloud storage infrastructure approved by Direct Manager and Director.",
            "details": {"contract_amount_usd": 11000.0, "approvals_obtained": ["manager", "director"], "vendor_quotes_count": 1},
            "original_policy_gold": "FLAG",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "FIN-POL-001 §2.1",
            "gold_chunk_id": "finance_expense_and_procurement_policy_chunk_2",
            "gold_policy_passage": "Tier 2 ($1,001 to $15,000 USD): Requires Direct Manager and Department Head (Director) approval."
        },
        {
            "id": "FIN-EDIT-007",
            "domain": "finance",
            "category": "procurement",
            "text": "Purchase of $14,800 USD development workstations with Manager and Director sign-offs.",
            "details": {"contract_amount_usd": 14800.0, "approvals_obtained": ["manager", "director"], "vendor_quotes_count": 1},
            "original_policy_gold": "FLAG",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "FIN-POL-001 §2.1",
            "gold_chunk_id": "finance_expense_and_procurement_policy_chunk_2",
            "gold_policy_passage": "Tier 2 ($1,001 to $15,000 USD): Requires Direct Manager and Department Head (Director) approval."
        },
        {
            "id": "FIN-EDIT-008",
            "domain": "finance",
            "category": "pcard",
            "text": "Employee submitting P-Card charge of $52 USD without itemized digital receipt on day 4.",
            "details": {"expense_type": "tools", "amount_usd": 52.0, "days_since_transaction": 4, "has_itemized_receipt": False},
            "original_policy_gold": "FLAG",
            "edited_policy_gold": "FLAG",
            "gold_decision": "FLAG",
            "gold_rule_id": "FIN-POL-001 §3.2",
            "gold_chunk_id": "finance_expense_and_procurement_policy_chunk_3",
            "gold_policy_passage": "Transactions exceeding $50 USD require an itemized digital tax invoice."
        },
        {
            "id": "FIN-EDIT-009",
            "domain": "finance",
            "category": "procurement",
            "text": "Purchase order of $16,500 USD with only Manager and Director approvals.",
            "details": {"contract_amount_usd": 16500.0, "approvals_obtained": ["manager", "director"], "vendor_quotes_count": 1},
            "original_policy_gold": "FLAG",
            "edited_policy_gold": "FLAG",
            "gold_decision": "FLAG",
            "gold_rule_id": "FIN-POL-001 §2.1",
            "gold_chunk_id": "finance_expense_and_procurement_policy_chunk_2",
            "gold_policy_passage": "Tier 3 ($15,001 to $50,000 USD): Requires Department Head, Procurement Lead, and Finance Controller approval."
        },
        {
            "id": "FIN-EDIT-010",
            "domain": "finance",
            "category": "procurement",
            "text": "Manager splitting $18,000 procurement into two $9,000 orders to avoid Controller review.",
            "details": {"contract_amount_usd": 18000.0, "split_invoicing": True},
            "original_policy_gold": "BLOCK",
            "edited_policy_gold": "BLOCK",
            "gold_decision": "BLOCK",
            "gold_rule_id": "FIN-POL-001 §2.2",
            "gold_chunk_id": "finance_expense_and_procurement_policy_chunk_2",
            "gold_policy_passage": "Intentionally splitting a single procurement order to circumvent approval limits is strictly prohibited."
        }
    ],
    "it_access": [
        {
            "id": "IT-EDIT-001",
            "domain": "it_access",
            "category": "infrastructure",
            "text": "Site reliability engineer requesting 6-hour break-glass emergency production DB access linked to active Jira P1 incident #INC-9042.",
            "details": {"is_break_glass": True, "duration_hours": 6, "has_incident_ticket": True},
            "original_policy_gold": "BLOCK",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "IT-SEC-001 §2.2",
            "gold_chunk_id": "it_access_access_control_security_policy_chunk_2",
            "gold_policy_passage": "Break-Glass Emergency Access: Maximum grant duration is 8 hours."
        },
        {
            "id": "IT-EDIT-002",
            "domain": "it_access",
            "category": "api_key_rotation",
            "text": "Security auditor reviewing service principal API key active for 75 days without rotation.",
            "details": {"days_since_rotation": 75},
            "original_policy_gold": "APPROVE",
            "edited_policy_gold": "FLAG",
            "gold_decision": "FLAG",
            "gold_rule_id": "IT-SEC-001 §4.1",
            "gold_chunk_id": "it_access_access_control_security_policy_chunk_4",
            "gold_policy_passage": "All machine credentials and API tokens must be stored in centralized Enterprise Vaults and rotated at least every 60 calendar days."
        },
        {
            "id": "IT-EDIT-003",
            "domain": "it_access",
            "category": "infrastructure",
            "text": "Engineer requesting 10-hour break-glass access with linked incident ticket.",
            "details": {"is_break_glass": True, "duration_hours": 10, "has_incident_ticket": True},
            "original_policy_gold": "BLOCK",
            "edited_policy_gold": "BLOCK",
            "gold_decision": "BLOCK",
            "gold_rule_id": "IT-SEC-001 §2.2",
            "gold_chunk_id": "it_access_access_control_security_policy_chunk_2",
            "gold_policy_passage": "Maximum grant duration is 8 hours, after which credentials automatically rotate."
        },
        {
            "id": "IT-EDIT-004",
            "domain": "it_access",
            "category": "infrastructure",
            "text": "DevOps engineer requesting 5-hour break-glass production server access with valid ServiceNow ticket.",
            "details": {"is_break_glass": True, "duration_hours": 5, "has_incident_ticket": True},
            "original_policy_gold": "BLOCK",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "IT-SEC-001 §2.2",
            "gold_chunk_id": "it_access_access_control_security_policy_chunk_2",
            "gold_policy_passage": "Break-Glass Emergency Access: Maximum grant duration is 8 hours."
        },
        {
            "id": "IT-EDIT-005",
            "domain": "it_access",
            "category": "infrastructure",
            "text": "Database administrator requesting 7.5-hour emergency production database access with incident ticket.",
            "details": {"is_break_glass": True, "duration_hours": 7.5, "has_incident_ticket": True},
            "original_policy_gold": "BLOCK",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "IT-SEC-001 §2.2",
            "gold_chunk_id": "it_access_access_control_security_policy_chunk_2",
            "gold_policy_passage": "Break-Glass Emergency Access: Maximum grant duration is 8 hours."
        },
        {
            "id": "IT-EDIT-006",
            "domain": "it_access",
            "category": "api_key_rotation",
            "text": "Audit scan detecting automated backup script API key active for 65 calendar days without rotation.",
            "details": {"days_since_rotation": 65},
            "original_policy_gold": "APPROVE",
            "edited_policy_gold": "FLAG",
            "gold_decision": "FLAG",
            "gold_rule_id": "IT-SEC-001 §4.1",
            "gold_chunk_id": "it_access_access_control_security_policy_chunk_4",
            "gold_policy_passage": "All machine credentials and API tokens must be rotated at least every 60 calendar days."
        },
        {
            "id": "IT-EDIT-007",
            "domain": "it_access",
            "category": "api_key_rotation",
            "text": "CI/CD deployment token active for 85 days without rotation in secret manager.",
            "details": {"days_since_rotation": 85},
            "original_policy_gold": "APPROVE",
            "edited_policy_gold": "FLAG",
            "gold_decision": "FLAG",
            "gold_rule_id": "IT-SEC-001 §4.1",
            "gold_chunk_id": "it_access_access_control_security_policy_chunk_4",
            "gold_policy_passage": "All machine credentials and API tokens must be rotated at least every 60 calendar days."
        },
        {
            "id": "IT-EDIT-008",
            "domain": "it_access",
            "category": "api_key_rotation",
            "text": "Service token active for 55 days within standard compliant rotation window.",
            "details": {"days_since_rotation": 55},
            "original_policy_gold": "APPROVE",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "IT-SEC-001 §4.1",
            "gold_chunk_id": "it_access_access_control_security_policy_chunk_4",
            "gold_policy_passage": "All machine credentials and API tokens must be rotated at least every 60 calendar days."
        },
        {
            "id": "IT-EDIT-009",
            "domain": "it_access",
            "category": "infrastructure",
            "text": "Incident commander requesting exactly 8-hour break-glass emergency production access with critical incident ticket.",
            "details": {"is_break_glass": True, "duration_hours": 8, "has_incident_ticket": True},
            "original_policy_gold": "BLOCK",
            "edited_policy_gold": "APPROVE",
            "gold_decision": "APPROVE",
            "gold_rule_id": "IT-SEC-001 §2.2",
            "gold_chunk_id": "it_access_access_control_security_policy_chunk_2",
            "gold_policy_passage": "Maximum grant duration is 8 hours, after which credentials automatically rotate."
        },
        {
            "id": "IT-EDIT-010",
            "domain": "it_access",
            "category": "separation_of_duties",
            "text": "Lead backend developer requesting direct release manager approval permissions for medical records system.",
            "details": {"has_deployment_privileges": True, "requests_approval_authority": True},
            "original_policy_gold": "BLOCK",
            "edited_policy_gold": "BLOCK",
            "gold_decision": "BLOCK",
            "gold_rule_id": "IT-SEC-001 §3.1",
            "gold_chunk_id": "it_access_access_control_security_policy_chunk_3",
            "gold_policy_passage": "No single individual may hold both software code deployment privileges and production release approval authority."
        }
    ]
}


def run_policy_edit_experiment() -> Dict[str, Any]:
    print("\n=======================================================")
    print(" RUNNING POLICY-CHANGE ROBUSTNESS EXPERIMENT")
    print("=======================================================")

    # 1. Rebuild FAISS indexes on edited policies
    build_all_indexes(force_rebuild=True, build_edited=True)

    results = {}
    all_sys_a_correct = []
    all_sys_b_correct = []
    all_sys_b_sources = []
    total_api_failures = 0

    for domain in DOMAINS:
        scenarios = POST_EDIT_SCENARIOS[domain]
        rule_fn = RULE_EVALUATORS[domain]
        judge_edited = RetrievalGovernanceJudge(domain=domain, is_edited=True)

        domain_res = {"domain": domain, "items": []}
        dom_a_correct = []
        dom_b_correct = []

        print(f"\n--- Domain: {domain.upper()} (Policy-Edit Test) ---")

        for sc in scenarios:
            # System A evaluates with STATIC unmodified rule code
            t0_a = time.perf_counter()
            pred_a = rule_fn(sc)
            latency_a = (time.perf_counter() - t0_a) * 1000

            gold_dec = sc["gold_decision"].upper().strip()
            a_correct = int(pred_a["decision"].upper().strip() == gold_dec)

            # System B evaluates with RE-INDEXED policy without code modifications
            pred_b = judge_edited.evaluate_scenario(scenario=sc, temperature=0.0, seed=42)
            b_correct = int(pred_b["decision"].upper().strip() == gold_dec)
            response_source = pred_b.get("response_source", "unknown")

            if response_source == "api_failure":
                total_api_failures += 1

            dom_a_correct.append(a_correct)
            dom_b_correct.append(b_correct)
            all_sys_a_correct.append(a_correct)
            all_sys_b_correct.append(b_correct)
            all_sys_b_sources.append(response_source)

            domain_res["items"].append({
                "scenario_id": sc["id"],
                "text": sc["text"],
                "original_gold": sc.get("original_policy_gold"),
                "new_gold": gold_dec,
                "system_a_pred": pred_a["decision"],
                "system_a_correct": a_correct,
                "system_a_latency_ms": round(latency_a, 4),
                "system_b_pred": pred_b["decision"],
                "system_b_correct": b_correct,
                "system_b_citation": pred_b.get("citation_text", "")[:100],
                "system_b_response_source": response_source,
                "system_b_model_used": pred_b.get("model_used", ""),
                "system_b_latency_ms": pred_b.get("latency_ms", 0.0),
            })

            source_tag = f" [{response_source}]" if response_source != "llm" else ""
            print(
                f"Scenario {sc['id']}: Gold={gold_dec} | "
                f"Sys A (Static Code)={pred_a['decision']} ({'CORRECT' if a_correct else 'STALE FAIL'}) | "
                f"Sys B (Retrieved)={pred_b['decision']} ({'CORRECT' if b_correct else 'FAIL'}){source_tag}"
            )

        domain_res["system_a_accuracy"] = compute_bootstrap_ci(dom_a_correct)
        domain_res["system_b_accuracy"] = compute_bootstrap_ci(dom_b_correct)
        results[domain] = domain_res

    # Overall McNemar Test comparing System A (stale code) vs System B (re-indexed)
    overall_mcnemar = mcnemar_test(all_sys_a_correct, all_sys_b_correct)
    overall_a_ci = compute_bootstrap_ci(all_sys_a_correct)
    overall_b_ci = compute_bootstrap_ci(all_sys_b_correct)

    # LLM-only System B accuracy (excluding API failures)
    llm_only_b_correct = [
        c for c, s in zip(all_sys_b_correct, all_sys_b_sources) if s == "llm"
    ]
    llm_only_b_ci = compute_bootstrap_ci(llm_only_b_correct) if llm_only_b_correct else None

    api_failure_rate = total_api_failures / len(all_sys_b_sources) if all_sys_b_sources else 0.0

    print("\n=======================================================")
    print(" POLICY-CHANGE ROBUSTNESS SUMMARY")
    print("=======================================================")
    print(f"System A (Static Rule Code) Post-Edit Accuracy: {overall_a_ci['mean']*100:.1f}% [95% CI: {overall_a_ci['ci_lower']*100:.1f}% - {overall_a_ci['ci_upper']*100:.1f}%]")
    print(f"System B (Zero-Code Re-indexed) Post-Edit Accuracy: {overall_b_ci['mean']*100:.1f}% [95% CI: {overall_b_ci['ci_lower']*100:.1f}% - {overall_b_ci['ci_upper']*100:.1f}%]")
    if llm_only_b_ci:
        print(f"System B LLM-Only Post-Edit Accuracy: {llm_only_b_ci['mean']*100:.1f}% (excluding {total_api_failures} API failures)")
    print(f"System B API Failure Rate: {api_failure_rate*100:.1f}%")
    print(f"McNemar Test: p-value = {overall_mcnemar['p_value']} (Significant: {overall_mcnemar['is_statistically_significant']})")
    print(f"System A Stale Error Rate: {(1.0 - overall_a_ci['mean'])*100:.1f}%")

    if api_failure_rate > 0.1:
        print(f"\n  *** WARNING: {api_failure_rate*100:.0f}% of System B responses came from API failures. ***")
        print(f"  *** Re-run with working API connection for valid robustness results. ***")

    out_payload = {
        "domain_results": results,
        "overall_system_a": overall_a_ci,
        "overall_system_b": overall_b_ci,
        "overall_system_b_llm_only": llm_only_b_ci,
        "mcnemar_test": overall_mcnemar,
        "api_failure_rate": round(api_failure_rate, 4),
        "total_api_failures": total_api_failures,
        "total_scenarios": len(all_sys_b_sources),
    }

    out_file = RESULTS_DIR / "policy_edit_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, default=str)

    print(f"Policy-edit experiment results saved to: {out_file}")
    return out_payload


if __name__ == "__main__":
    run_policy_edit_experiment()
