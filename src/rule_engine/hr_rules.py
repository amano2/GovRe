"""System A: Rule-Based Governance Engine for HR Operations Domain.
Hand-encoded rules faithful to HR-POL-001 (Leave), HR-POL-002 (Reimbursements), and HR-POL-004 (Code of Conduct).
"""

from typing import Dict, Any

def evaluate_hr_rule(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates an HR request scenario against explicit rule logic.
    Returns:
        decision: "APPROVE" | "FLAG" | "BLOCK"
        rule_id: The specific policy rule ID cited
        reason: Explanation of the rule decision
    """
    category = scenario.get("category", "").lower()
    details = scenario.get("details", {})
    text = scenario.get("text", "").lower()

    # ==========================
    # 1. LEAVE & TIME-OFF RULES
    # ==========================
    if category in ["leave", "time_off"] or "leave" in text or "vacation" in text or "pto" in text:
        leave_type = details.get("leave_type", "").lower()
        days_requested = details.get("days_requested", 0)
        notice_days = details.get("notice_days", 0)
        medical_cert_provided = details.get("medical_certificate_provided", None)
        service_days = details.get("service_days_in_past_year", 365)
        comp_off_days_since_worked = details.get("comp_off_days_since_worked", 0)
        has_prior_preapproval = details.get("manager_preapproved", False)
        relation = details.get("bereavement_relation", "").lower()

        # 1.1 Annual Leave / PTO
        if "annual" in leave_type or "pto" in leave_type or "vacation" in leave_type or "earned" in leave_type:
            if days_requested > 20:
                return {
                    "decision": "BLOCK",
                    "rule_id": "HR-POL-001 §2.1",
                    "reason": f"Requested {days_requested} days exceeds maximum annual PTO accrual limit of 20 days per year."
                }
            if days_requested > 3 and notice_days < 14:
                return {
                    "decision": "FLAG",
                    "rule_id": "HR-POL-001 §2.1",
                    "reason": f"Planned leave exceeding 3 consecutive days ({days_requested} days) requires at least 2 weeks (14 days) advance notice; received {notice_days} days."
                }
            if days_requested <= 2 and notice_days < 2 and "emergency" not in text:
                return {
                    "decision": "FLAG",
                    "rule_id": "HR-POL-001 §3.2",
                    "reason": "Planned leave of 1-2 days must be submitted at least 48 hours (2 days) in advance."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "HR-POL-001 §2.1",
                "reason": "Annual leave request is within annual entitlement and complies with notice requirements."
            }

        # 1.2 Sick Leave
        if "sick" in leave_type or "medical" in leave_type:
            if days_requested > 10:
                return {
                    "decision": "BLOCK",
                    "rule_id": "HR-POL-001 §2.2",
                    "reason": f"Sick leave request of {days_requested} days exceeds the annual 10-day entitlement."
                }
            if days_requested >= 3 and medical_cert_provided is False:
                return {
                    "decision": "FLAG",
                    "rule_id": "HR-POL-001 §2.2",
                    "reason": f"Medical absence of {days_requested} consecutive business days requires an official medical certificate uploaded within 48 hours."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "HR-POL-001 §2.2",
                "reason": "Sick leave request is within 10-day entitlement and complies with medical certification rules."
            }

        # 1.3 Parental Leave (Maternity / Paternity / Adoption)
        if "maternity" in leave_type:
            if service_days < 80:
                return {
                    "decision": "BLOCK",
                    "rule_id": "HR-POL-001 §2.3",
                    "reason": f"Maternity leave requires a minimum of 80 days of service in preceding 12 months; employee has {service_days} days."
                }
            if days_requested > 182:  # 26 weeks
                return {
                    "decision": "BLOCK",
                    "rule_id": "HR-POL-001 §2.3",
                    "reason": f"Maternity leave requested ({days_requested} days) exceeds statutory 26-week (182 calendar days) entitlement."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "HR-POL-001 §2.3",
                "reason": "Maternity leave request fulfills 80 days service requirement and is within 26-week limit."
            }

        if "paternity" in leave_type:
            if days_requested > 20:  # 4 weeks
                return {
                    "decision": "BLOCK",
                    "rule_id": "HR-POL-001 §2.3",
                    "reason": f"Paternity leave requested ({days_requested} working days) exceeds 4-week (20 working days) limit."
                }
            months_since_birth = details.get("months_since_birth", 1)
            if months_since_birth > 6:
                return {
                    "decision": "BLOCK",
                    "rule_id": "HR-POL-001 §2.3",
                    "reason": "Paternity leave must be availed within 6 months of childbirth or adoption."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "HR-POL-001 §2.3",
                "reason": "Paternity leave request is within 4-week limit and within 6-month validity window."
            }

        # 1.4 Bereavement Leave
        if "bereavement" in leave_type or "demise" in text or "funeral" in text:
            immediate = ["spouse", "child", "parent", "father", "mother", "sibling", "brother", "sister", "parent-in-law"]
            if any(rel in relation for rel in immediate) or any(rel in text for rel in immediate):
                if days_requested > 5:
                    return {
                        "decision": "BLOCK",
                        "rule_id": "HR-POL-001 §2.4",
                        "reason": f"Bereavement leave for immediate family is capped at 5 consecutive business days; requested {days_requested}."
                    }
                return {
                    "decision": "APPROVE",
                    "rule_id": "HR-POL-001 §2.4",
                    "reason": "Bereavement leave for immediate family is within 5-day entitlement."
                }
            else:
                if days_requested > 2:
                    return {
                        "decision": "BLOCK",
                        "rule_id": "HR-POL-001 §2.4",
                        "reason": f"Bereavement leave for extended family is capped at 2 consecutive business days; requested {days_requested}."
                    }
                return {
                    "decision": "APPROVE",
                    "rule_id": "HR-POL-001 §2.4",
                    "reason": "Bereavement leave for extended family is within 2-day entitlement."
                }

        # 1.5 Comp-Off
        if "comp" in leave_type:
            if not has_prior_preapproval:
                return {
                    "decision": "BLOCK",
                    "rule_id": "HR-POL-001 §2.5",
                    "reason": "Comp-off claims require prior manager pre-approval before working on holiday or weekend."
                }
            if comp_off_days_since_worked > 60:
                return {
                    "decision": "BLOCK",
                    "rule_id": "HR-POL-001 §2.5",
                    "reason": f"Comp-off expired; must be utilized within 60 calendar days (claimed at {comp_off_days_since_worked} days)."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "HR-POL-001 §2.5",
                "reason": "Comp-off claim is pre-approved and within 60-day utilization window."
            }

    # ==================================
    # 2. REIMBURSEMENT & EXPENSE RULES
    # ==================================
    if category in ["reimbursement", "expense"] or "claim" in text or "reimburs" in text:
        exp_type = details.get("expense_type", "").lower()
        amount = details.get("amount_usd", 0.0)
        days_since_expense = details.get("days_since_expense", 0)
        has_receipt = details.get("has_itemized_receipt", True)
        has_preapproval = details.get("has_preapproval", False)
        city_tier = details.get("city_tier", 2)
        flight_duration_hours = details.get("flight_duration_hours", 0)
        travel_class = details.get("travel_class", "economy").lower()
        tenure_days = details.get("tenure_days", 30)

        # 2.1 Submission Window & Receipt Threshold
        if days_since_expense > 60:
            return {
                "decision": "BLOCK",
                "rule_id": "HR-POL-002 §5.1",
                "reason": f"Expense claim submitted after {days_since_expense} days; claims older than 60 days are automatically rejected."
            }
        if days_since_expense > 30:
            return {
                "decision": "FLAG",
                "rule_id": "HR-POL-002 §5.1",
                "reason": f"Expense claim submitted after {days_since_expense} days (standard filing window is 30 calendar days)."
            }
        if amount > 15 and not has_receipt:
            return {
                "decision": "FLAG",
                "rule_id": "HR-POL-002 §5.2",
                "reason": f"Original digital/scanned receipt is mandatory for transactions exceeding $15 USD (amount: ${amount})."
            }

        # 2.2 Broadband / Remote Internet
        if "broadband" in exp_type or "internet" in exp_type or "wifi" in text:
            if amount > 50:
                return {
                    "decision": "BLOCK",
                    "rule_id": "HR-POL-002 §2.1",
                    "reason": f"Home internet allowance is capped at $50 USD per month; claimed ${amount}."
                }
            if not has_receipt:
                return {
                    "decision": "FLAG",
                    "rule_id": "HR-POL-002 §2.1",
                    "reason": "Itemized billing statement showing employee name and address is required for broadband reimbursement."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "HR-POL-002 §2.1",
                "reason": "Home broadband claim is within the $50 monthly limit with itemized billing."
            }

        # 2.3 Ergonomic Setup
        if "ergo" in exp_type or "home office" in exp_type or "chair" in text or "desk" in text:
            if tenure_days > 90:
                return {
                    "decision": "BLOCK",
                    "rule_id": "HR-POL-002 §2.2",
                    "reason": f"Ergonomic setup stipend must be claimed within the first 90 days of employment; tenure is {tenure_days} days."
                }
            if amount > 300:
                return {
                    "decision": "BLOCK",
                    "rule_id": "HR-POL-002 §2.2",
                    "reason": f"Ergonomic home office stipend is capped at $300 USD one-time; claimed ${amount}."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "HR-POL-002 §2.2",
                "reason": "Ergonomic setup claim is within $300 limit and filed within 90 days of hire."
            }

        # 2.4 Airfare
        if "air" in exp_type or "flight" in exp_type:
            if flight_duration_hours < 6 and travel_class not in ["economy", "coach"]:
                return {
                    "decision": "BLOCK",
                    "rule_id": "HR-POL-002 §3.1",
                    "reason": f"Flights under 6 hours ({flight_duration_hours}h) require mandatory Economy class; requested {travel_class}."
                }
            if flight_duration_hours >= 6 and travel_class in ["business", "first"]:
                return {
                    "decision": "BLOCK",
                    "rule_id": "HR-POL-002 §3.1",
                    "reason": "Business/First class is not permissible under standard travel policy; only Premium Economy allowed for flights over 6h with approval."
                }
            if flight_duration_hours >= 6 and "premium" in travel_class and not has_preapproval:
                return {
                    "decision": "FLAG",
                    "rule_id": "HR-POL-002 §3.1",
                    "reason": "Premium Economy for flights over 6 hours requires prior Department Head written approval."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "HR-POL-002 §3.1",
                "reason": "Airfare booking adheres to class and duration guidelines."
            }

        # 2.5 Meals / Per Diem
        if "meal" in exp_type or "food" in exp_type or "dinner" in text or "lunch" in text:
            if "alcohol" in text or details.get("includes_alcohol", False):
                if not details.get("director_preapproved", False):
                    return {
                        "decision": "BLOCK",
                        "rule_id": "HR-POL-002 §3.2",
                        "reason": "Alcohol is strictly non-reimbursable unless part of an authorized client entertainment dinner with Director-level pre-approval."
                    }
            max_meal = 75 if city_tier == 1 else 50
            if amount > max_meal:
                return {
                    "decision": "BLOCK",
                    "rule_id": "HR-POL-002 §3.2",
                    "reason": f"Daily meal expense of ${amount} exceeds Tier {city_tier} per diem limit of ${max_meal} USD per day."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "HR-POL-002 §3.2",
                "reason": f"Meal expense of ${amount} is within Tier {city_tier} per diem allowance."
            }

        # 2.6 Lodging
        if "hotel" in exp_type or "lodging" in exp_type or "stay" in text:
            max_hotel = 180 if city_tier == 1 else 120
            if amount > max_hotel:
                return {
                    "decision": "BLOCK",
                    "rule_id": "HR-POL-002 §3.3",
                    "reason": f"Hotel rate of ${amount}/night exceeds Tier {city_tier} ceiling cap of ${max_hotel} USD per night."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "HR-POL-002 §3.3",
                "reason": f"Hotel booking is within Tier {city_tier} ceiling cap."
            }

        # 2.7 Learning & Development
        if "cert" in exp_type or "learning" in exp_type or "course" in exp_type or "training" in exp_type:
            if amount > 1000:
                return {
                    "decision": "BLOCK",
                    "rule_id": "HR-POL-002 §4.1",
                    "reason": f"L&D claim of ${amount} exceeds annual budget cap of $1,000 USD per calendar year."
                }
            if not has_preapproval:
                return {
                    "decision": "FLAG",
                    "rule_id": "HR-POL-002 §4.1",
                    "reason": "Pre-approval from direct manager is required before purchasing courses or certifications."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "HR-POL-002 §4.1",
                "reason": "L&D expense is within $1,000 budget and pre-approved."
            }

    # =================================
    # 3. CODE OF CONDUCT & ETHICS RULES
    # =================================
    if category in ["ethics", "conduct", "compliance"] or "gift" in text or "moonlight" in text or "social media" in text:
        gift_val = details.get("gift_value_usd", 0)
        is_cash = details.get("is_cash_or_gift_card", False)
        is_commercial = details.get("is_commercial_employment", False)
        has_vp_hr_approval = details.get("vp_and_hr_approved", False)
        involves_public_ai_tool = details.get("uploads_proprietary_data_to_public_ai", False)

        if is_cash:
            return {
                "decision": "BLOCK",
                "rule_id": "HR-POL-004 §2.2",
                "reason": "Cash or cash equivalents (gift cards, prepaid debit cards) of any amount are strictly prohibited."
            }
        if gift_val > 50:
            if not details.get("disclosed_to_compliance", False):
                return {
                    "decision": "FLAG",
                    "rule_id": "HR-POL-004 §2.2",
                    "reason": f"Accepting personal gifts exceeding $50 USD (value: ${gift_val}) requires mandatory disclosure to the Compliance Team."
                }
        if is_commercial:
            return {
                "decision": "BLOCK",
                "rule_id": "HR-POL-004 §2.1",
                "reason": "Full-time employees are strictly prohibited from engaging in commercial secondary employment or consulting."
            }
        if details.get("is_volunteer_or_academic", False) and not has_vp_hr_approval:
            return {
                "decision": "FLAG",
                "rule_id": "HR-POL-004 §2.1",
                "reason": "Non-commercial volunteer or academic lecturing requires prior written disclosure and approval from Department VP and HR."
            }
        if involves_public_ai_tool:
            return {
                "decision": "BLOCK",
                "rule_id": "HR-POL-004 §3.1",
                "reason": "Uploading proprietary source code or PII to unapproved public cloud or unauthorized consumer AI tools is strictly prohibited."
            }
        return {
            "decision": "APPROVE",
            "rule_id": "HR-POL-004 §1.1",
            "reason": "Activity complies with professional integrity and ethics guidelines."
        }

    # Default Fallback for unspecified requests
    return {
        "decision": "FLAG",
        "rule_id": "HR-POL-001 §1.0",
        "reason": "Request does not map cleanly to standard automated approval criteria; flagged for human HR specialist review."
    }
