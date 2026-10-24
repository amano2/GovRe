import time
from typing import Dict, Any

from src.rule_engine import evaluate_hr_rule, evaluate_finance_rule, evaluate_it_access_rule
from src.retrieval_governance import RetrievalGovernanceJudge

class HybridGovernanceRouter:
    """
    Tiered Hybrid Router for Governance Grounding.
    
    Tier 1: Fast deterministic rule engine (System A) for static arithmetic checks.
    Tier 2: Dynamic retrieval-grounded judge (System B) for complex, qualitative, or evolving policies.
    """
    
    def __init__(self, domain: str, judge: RetrievalGovernanceJudge = None):
        self.domain = domain
        self.judge = judge or RetrievalGovernanceJudge(domain=domain, is_edited=False)
        self.rule_evaluators = {
            "hr": evaluate_hr_rule,
            "finance": evaluate_finance_rule,
            "it_access": evaluate_it_access_rule
        }
        
        # Define categories that are predominantly arithmetic/deterministic thresholds
        self.arithmetic_categories = {
            "hr": ["leave", "reimbursement"],
            "finance": ["procurement", "pcard", "entertainment"],
            "it_access": ["infrastructure", "api_key_rotation"]
        }

    def evaluate_scenario(self, scenario: Dict[str, Any], temperature: float = 0.0, seed: int = 42) -> Dict[str, Any]:
        t0 = time.perf_counter()
        
        category = scenario.get("category", "")
        
        # Tier 1 Filter: Arithmetic parameters and explicitly covered by rules
        if category in self.arithmetic_categories.get(self.domain, []):
            rule_fn = self.rule_evaluators[self.domain]
            rule_result = rule_fn(scenario)
            
            # Check if System A returned a fallback/unhandled case
            reason_lower = rule_result.get("reason", "").lower()
            is_fallback = "manual" in reason_lower or "review" in reason_lower or "specialist" in reason_lower or "does not map cleanly" in reason_lower
            
            if not is_fallback:
                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000
                return {
                    "decision": rule_result["decision"],
                    "citation_chunk_id": "",
                    "citation_text": rule_result.get("rule_id", ""),
                    "reasoning": f"[System A Routed] {rule_result.get('reason', '')}",
                    "response_source": "system_a",
                    "latency_ms": round(latency_ms, 4),
                    "model_used": "rule_engine_tier1"
                }
                
        # Tier 2 Fallback: Route to System B (Retrieval-Grounded Judge)
        b_result = self.judge.evaluate_scenario(scenario, temperature=temperature, seed=seed)
        t1 = time.perf_counter()
        
        # Prefix the source and reasoning to clearly identify the routing path
        original_source = b_result.get("response_source", "unknown")
        b_result["response_source"] = f"system_b_{original_source}"
        b_result["reasoning"] = f"[System B Routed] {b_result.get('reasoning', '')}"
        
        # Override latency to include router overhead (negligible, but accurate)
        b_result["latency_ms"] = round((t1 - t0) * 1000, 4)
        
        return b_result
