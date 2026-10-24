"""System A: Rule-Based Governance Engine for Expense & Corporate Finance Domain.
Hand-encoded rules faithful to FIN-POL-001 (Finance & Procurement Policy).
"""

from typing import Dict, Any

def evaluate_finance_rule(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a finance/procurement request scenario against explicit rule logic.
    Returns:
        decision: "APPROVE" | "FLAG" | "BLOCK"
        rule_id: The specific policy rule ID cited
        reason: Explanation of the rule decision
    """
    category = scenario.get("category", "").lower()
    details = scenario.get("details", {})
    text = scenario.get("text", "").lower()

    # ====================================================
    # 1. SPLIT INVOICING & FRAUD CHECK
    # ====================================================
    if details.get("is_split_invoice", False) or "split invoice" in text or "circumvent approval" in text:
        return {
            "decision": "BLOCK",
            "rule_id": "FIN-POL-001 §2.2",
            "reason": "Intentionally splitting purchase orders or invoices to circumvent approval tiers is strictly prohibited."
        }

    # ====================================================
    # 2. P-CARD GOVERNANCE & PERMISSIBLE EXPENSES
    # ====================================================
    if category in ["pcard", "credit_card", "card"] or "p-card" in text or "corporate card" in text:
        expense_type = details.get("expense_type", "").lower()
        amount = details.get("amount_usd", 0.0)
        days_since_tx = details.get("days_since_transaction", 0)
        has_receipt = details.get("has_itemized_receipt", True)
        is_personal = details.get("is_personal_expense", False)

        prohibited_types = ["crypto", "cryptocurrency", "gambling", "luxury", "spa", "personal cash advance"]
        if is_personal or any(pt in expense_type for pt in prohibited_types) or any(pt in text for pt in prohibited_types):
            return {
                "decision": "BLOCK",
                "rule_id": "FIN-POL-001 §3.1",
                "reason": "Corporate P-Cards must never be used for personal expenses, cash advances, cryptocurrency, gambling, or luxury services."
            }
        if days_since_tx > 30:
            return {
                "decision": "BLOCK",
                "rule_id": "FIN-POL-001 §3.2",
                "reason": f"P-Card reconciliation overdue ({days_since_tx} days); unreconciled balances beyond 30 days trigger card suspension and payroll deduction."
            }
        if amount > 25 and not has_receipt:
            return {
                "decision": "FLAG",
                "rule_id": "FIN-POL-001 §3.2",
                "reason": f"Itemized tax invoice is mandatory for P-Card transactions exceeding $25 USD (amount: ${amount})."
            }
        if days_since_tx > 7 and not has_receipt:
            return {
                "decision": "FLAG",
                "rule_id": "FIN-POL-001 §3.2",
                "reason": f"P-Card receipts must be reconciled within 7 business days of posting (currently {days_since_tx} days)."
            }
        return {
            "decision": "APPROVE",
            "rule_id": "FIN-POL-001 §3.1",
            "reason": "Corporate card transaction is for permissible business expense and properly reconciled."
        }

    # ====================================================
    # 3. PROCUREMENT & CONTRACT AUTHORIZATION MATRIX
    # ====================================================
    if category in ["procurement", "purchase_order", "po", "contract"] or "purchase" in text or "contract" in text or "vendor order" in text:
        amount = details.get("contract_amount_usd", details.get("amount_usd", 0.0))
        approvals = [a.lower() for a in details.get("approvals_obtained", [])]
        quote_count = details.get("vendor_quotes_count", 1)
        is_hardware = details.get("is_it_hardware", False) or "laptop" in text or "server" in text or "monitor" in text

        # 3.1 IT Hardware restriction
        if is_hardware and details.get("purchased_via_reimbursement", False):
            return {
                "decision": "BLOCK",
                "rule_id": "FIN-POL-001 §6.1",
                "reason": "Hardware and computing devices must be procured exclusively via Centralized IT Procurement, not employee reimbursement."
            }

        # 3.2 Tier 4 (> $50,000)
        if amount > 50000:
            has_cfo = "cfo" in approvals or "finance_controller" in approvals
            has_ceo = "ceo" in approvals or "board" in approvals
            if quote_count < 3:
                return {
                    "decision": "FLAG",
                    "rule_id": "FIN-POL-001 §2.1",
                    "reason": f"Procurements exceeding $50,000 require minimum 3 competitive vendor quotes (provided: {quote_count})."
                }
            if not (has_cfo and has_ceo):
                return {
                    "decision": "FLAG",
                    "rule_id": "FIN-POL-001 §2.1",
                    "reason": "Tier 4 purchases exceeding $50,000 USD require explicit sign-off from both CFO and CEO."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "FIN-POL-001 §2.1",
                "reason": "Tier 4 procurement meets CFO/CEO approval and 3 competitive bids requirement."
            }

        # 3.3 Tier 3 ($10,001 - $50,000)
        if amount > 10000:
            has_dept_head = "director" in approvals or "dept_head" in approvals or "vp" in approvals
            has_procurement = "procurement" in approvals or "procurement_lead" in approvals
            has_controller = "controller" in approvals or "finance_controller" in approvals or "cfo" in approvals
            if not (has_dept_head and has_procurement and has_controller):
                return {
                    "decision": "FLAG",
                    "rule_id": "FIN-POL-001 §2.1",
                    "reason": f"Tier 3 purchase (${amount}) requires approvals from Department Head, Procurement Lead, and Finance Controller."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "FIN-POL-001 §2.1",
                "reason": "Tier 3 procurement fulfills all required departmental and controller sign-offs."
            }

        # 3.4 Tier 2 ($1,001 - $10,000)
        if amount > 1000:
            has_manager = "manager" in approvals or "direct_manager" in approvals
            has_dept_head = "director" in approvals or "dept_head" in approvals or "vp" in approvals
            if not (has_manager and has_dept_head):
                return {
                    "decision": "FLAG",
                    "rule_id": "FIN-POL-001 §2.1",
                    "reason": f"Tier 2 purchase (${amount}) requires approvals from both Direct Manager and Department Head (Director)."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "FIN-POL-001 §2.1",
                "reason": "Tier 2 procurement approved by Direct Manager and Department Head."
            }

        # 3.5 Tier 1 (<= $1,000)
        has_manager = "manager" in approvals or "direct_manager" in approvals
        if not has_manager:
            return {
                "decision": "FLAG",
                "rule_id": "FIN-POL-001 §2.1",
                "reason": f"Tier 1 purchase (${amount}) requires Direct Manager approval."
            }
        return {
            "decision": "APPROVE",
            "rule_id": "FIN-POL-001 §2.1",
            "reason": "Tier 1 purchase within $1,000 threshold with manager approval."
        }

    # ====================================================
    # 4. VENDOR ADVANCES & SCREENING
    # ====================================================
    if category in ["vendor_onboarding", "advance_payment"] or "vendor advance" in text or "vendor screening" in text:
        is_screened = details.get("kyv_sanctions_completed", True)
        has_legal_override = details.get("legal_head_approved", False)
        advance_pct = details.get("advance_payment_percentage", 0)
        has_bank_guarantee = details.get("has_bank_guarantee", False)
        is_cots_saas = details.get("is_standard_cots_saas", False)
        contract_amt = details.get("contract_amount_usd", 0)

        if not is_screened and not has_legal_override:
            return {
                "decision": "BLOCK",
                "rule_id": "FIN-POL-001 §4.1",
                "reason": "New vendors must complete KYV and sanctions screening prior to contract execution or initial payment."
            }
        if not is_screened and has_legal_override:
            return {
                "decision": "FLAG",
                "rule_id": "FIN-POL-001 §4.1",
                "reason": "Emergency vendor onboarding without standard KYV screening authorized via Legal & Compliance override."
            }
        if advance_pct > 30 and not (has_bank_guarantee or (is_cots_saas and contract_amt <= 25000)):
            return {
                "decision": "BLOCK",
                "rule_id": "FIN-POL-001 §4.2",
                "reason": f"Advance supplier payments exceeding 30% (requested: {advance_pct}%) are prohibited without a performance bank guarantee."
            }
        return {
            "decision": "APPROVE",
            "rule_id": "FIN-POL-001 §4.1",
            "reason": "Vendor onboarding and advance payment terms meet risk management guidelines."
        }

    # ====================================================
    # 5. CLIENT ENTERTAINMENT & TEAM EVENTS
    # ====================================================
    if category in ["entertainment", "client_dinner", "team_event"] or "client dinner" in text or "team dinner" in text:
        is_client = details.get("is_client_event", True)
        cost_per_person = details.get("cost_per_person_usd", 0.0)
        total_event_cost = details.get("total_event_cost_usd", 0.0)
        attendees_listed = details.get("attendees_listed", True)
        has_vp_approval = details.get("vp_preapproved", False)
        has_manager_approval = details.get("manager_approved", False)

        if is_client:
            if cost_per_person > 100:
                return {
                    "decision": "BLOCK",
                    "rule_id": "FIN-POL-001 §5.1",
                    "reason": f"Client dining expense (${cost_per_person}/attendee) exceeds the maximum limit of $100 USD per attendee."
                }
            if not attendees_listed:
                return {
                    "decision": "FLAG",
                    "rule_id": "FIN-POL-001 §5.1",
                    "reason": "Itemized dining receipt must list attendee names, affiliations, and explicit business purpose."
                }
            if total_event_cost > 1000 and not has_vp_approval:
                return {
                    "decision": "FLAG",
                    "rule_id": "FIN-POL-001 §5.1",
                    "reason": f"Single client entertainment event exceeding $1,000 USD (${total_event_cost}) requires advance Department VP written authorization."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "FIN-POL-001 §5.1",
                "reason": "Client entertainment complies with per-attendee limit and documentation requirements."
            }
        else:
            # Team event
            if cost_per_person > 40:
                return {
                    "decision": "BLOCK",
                    "rule_id": "FIN-POL-001 §5.2",
                    "reason": f"Internal team dinner cost (${cost_per_person}/person) exceeds the $40 USD per employee ceiling."
                }
            if total_event_cost > 500 and not has_manager_approval:
                return {
                    "decision": "FLAG",
                    "rule_id": "FIN-POL-001 §5.2",
                    "reason": f"Team celebration exceeding $500 USD (${total_event_cost}) requires prior manager approval."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "FIN-POL-001 §5.2",
                "reason": "Internal team event is within $40/head limit and appropriately authorized."
            }

    # Default Fallback
    return {
        "decision": "FLAG",
        "rule_id": "FIN-POL-001 §1.0",
        "reason": "Request requires manual financial compliance review."
    }
