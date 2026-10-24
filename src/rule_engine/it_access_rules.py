"""System A: Rule-Based Governance Engine for IT Access Control & Security Domain.
Hand-encoded rules faithful to IT-SEC-001 (Access Control & Information Security Policy).
"""

from typing import Dict, Any

def evaluate_it_access_rule(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates an IT access or security request scenario against explicit rule logic.
    Returns:
        decision: "APPROVE" | "FLAG" | "BLOCK"
        rule_id: The specific policy rule ID cited
        reason: Explanation of the rule decision
    """
    category = scenario.get("category", "").lower()
    details = scenario.get("details", {})
    text = scenario.get("text", "").lower()

    # ====================================================
    # 1. SECRET MANAGEMENT & SOURCE CODE SECURITY
    # ====================================================
    if details.get("is_hardcoded_in_repo", False) or "hardcode" in text or "commit secret" in text:
        return {
            "decision": "BLOCK",
            "rule_id": "IT-SEC-001 §4.1",
            "reason": "Hardcoding API keys, database strings, or credentials into source code repositories is a critical security violation."
        }

    # ====================================================
    # 2. DATA CLASSIFICATION & EXFILTRATION
    # ====================================================
    if category in ["data_export", "data_access", "storage"] or "pii" in text or "customer data" in text or "usb" in text:
        data_tier = details.get("data_classification", "internal").lower()
        has_dpo_approval = details.get("dpo_approved", False)
        is_encrypted = details.get("is_encrypted", True)
        target_device = details.get("target_device_or_destination", "").lower()

        if "usb" in target_device or "personal" in target_device or "unencrypted" in target_device:
            return {
                "decision": "BLOCK",
                "rule_id": "IT-SEC-001 §5.2",
                "reason": "Downloading customer PII or enterprise data to unencrypted USB drives or personal devices is strictly prohibited."
            }

        if data_tier in ["restricted", "pii", "phi", "sensitive"]:
            if not has_dpo_approval:
                return {
                    "decision": "BLOCK",
                    "rule_id": "IT-SEC-001 §5.1",
                    "reason": "Access to Restricted / Highly Sensitive PII/PHI data requires explicit Data Protection Officer (DPO) authorization."
                }
            if not is_encrypted:
                return {
                    "decision": "BLOCK",
                    "rule_id": "IT-SEC-001 §5.1",
                    "reason": "Restricted data access must be encrypted at rest and in transit."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "IT-SEC-001 §5.1",
                "reason": "Restricted data access has DPO authorization and mandatory encryption."
            }

    # ====================================================
    # 3. PRODUCTION & BREAK-GLASS ACCESS
    # ====================================================
    if category in ["production_access", "database_access", "infrastructure"] or "production" in text or "prod db" in text:
        is_standing_access = details.get("is_standing_access", False)
        is_break_glass = details.get("is_break_glass", False)
        duration_hours = details.get("duration_hours", 0)
        has_incident_ticket = details.get("has_incident_ticket", False)
        approvals = [a.lower() for a in details.get("approvals_obtained", [])]

        if is_standing_access:
            return {
                "decision": "BLOCK",
                "rule_id": "IT-SEC-001 §2.2",
                "reason": "Direct permanent / standing read-write access to production environments is strictly prohibited."
            }

        if is_break_glass or "break-glass" in text or "emergency" in text:
            if not has_incident_ticket:
                return {
                    "decision": "BLOCK",
                    "rule_id": "IT-SEC-001 §2.2",
                    "reason": "Break-glass emergency production access requires a linked incident or change ticket."
                }
            if duration_hours > 4:
                return {
                    "decision": "BLOCK",
                    "rule_id": "IT-SEC-001 §2.2",
                    "reason": f"Break-glass access duration ({duration_hours} hours) exceeds maximum allowed grant of 4 hours."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "IT-SEC-001 §2.2",
                "reason": "Break-glass JIT production access complies with 4-hour limit and has linked incident ticket."
            }

        # Standard non-prod or application access
        has_manager = "manager" in approvals
        has_app_owner = "app_owner" in approvals or "system_owner" in approvals
        if not (has_manager and has_app_owner):
            return {
                "decision": "FLAG",
                "rule_id": "IT-SEC-001 §2.1",
                "reason": "Non-standard application access requires both direct manager approval and application system owner sign-off."
            }
        return {
            "decision": "APPROVE",
            "rule_id": "IT-SEC-001 §2.1",
            "reason": "Access request satisfies RBAC profile and system owner sign-off."
        }

    # ====================================================
    # 4. SEPARATION OF DUTIES (SOD) & MFA
    # ====================================================
    if category in ["sod", "privileged_access", "auth", "mfa"] or "separation of duties" in text or "mfa" in text or "admin" in text:
        has_dev_and_release = details.get("holds_both_deploy_and_approval", False)
        modifies_siem_logs = details.get("requests_siem_delete_access", False)
        mfa_type = details.get("mfa_type", "").lower()
        is_privileged_role = details.get("is_privileged_role", False)

        if has_dev_and_release:
            return {
                "decision": "BLOCK",
                "rule_id": "IT-SEC-001 §3.1",
                "reason": "Separation of Duties violation: no single individual may hold both code deployment and production release approval authority."
            }
        if modifies_siem_logs:
            return {
                "decision": "BLOCK",
                "rule_id": "IT-SEC-001 §3.1",
                "reason": "Developers are prohibited from possessing write/delete permissions in production audit or SIEM logging repositories."
            }
        if is_privileged_role and ("sms" in mfa_type or "none" in mfa_type):
            return {
                "decision": "BLOCK",
                "rule_id": "IT-SEC-001 §3.2",
                "reason": "Privileged/IAM administrator accounts require FIDO2/Hardware Keys or Phishing-Resistant MFA; SMS OTP is prohibited."
            }
        return {
            "decision": "APPROVE",
            "rule_id": "IT-SEC-001 §3.1",
            "reason": "Privileged role allocation complies with Separation of Duties and MFA security standards."
        }

    # ====================================================
    # 5. THIRD PARTY INTEGRATION & SECRET ROTATION
    # ====================================================
    if category in ["third_party_integration", "api_key_rotation"] or "third party" in text or "secret rotation" in text:
        has_ciso_review = details.get("ciso_reviewed", False)
        has_signed_dpa = details.get("signed_dpa", False)
        days_since_rotation = details.get("days_since_rotation", 0)

        if details.get("is_third_party_integration", False):
            if not (has_ciso_review and has_signed_dpa):
                return {
                    "decision": "FLAG",
                    "rule_id": "IT-SEC-001 §4.2",
                    "reason": "Third-party SaaS/AI data warehouse integration requires CISO architecture review and signed Data Processing Agreement (DPA)."
                }
            return {
                "decision": "APPROVE",
                "rule_id": "IT-SEC-001 §4.2",
                "reason": "Third-party integration is CISO-approved with signed DPA."
            }

        if days_since_rotation > 90:
            return {
                "decision": "FLAG",
                "rule_id": "IT-SEC-001 §4.1",
                "reason": f"API secret has not been rotated in {days_since_rotation} days (mandatory rotation SLA is 90 calendar days)."
            }
        return {
            "decision": "APPROVE",
            "rule_id": "IT-SEC-001 §4.1",
            "reason": "Secret rotation is within the 90-day compliance lifecycle."
        }

    # Default Fallback
    return {
        "decision": "FLAG",
        "rule_id": "IT-SEC-001 §1.0",
        "reason": "Request requires IT Information Security operational review."
    }
