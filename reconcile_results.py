import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR
from stats.significance_tests import compute_bootstrap_ci, mcnemar_test
from eval.run_eval import compute_effect_size

def update_main_eval():
    file_path = RESULTS_DIR / "main_eval_results.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for domain, res in data.items():
        # Update System A stats
        a_items = res["system_a"].get("items", [])
        if a_items:
            a_acc_outcomes = [item["accuracy_correct"] for item in a_items]
            a_exp_outcomes = [1 if item.get("explainability", {}).get("citation_correct") else 0 for item in a_items]
            res["system_a"]["accuracy_ci"] = compute_bootstrap_ci(a_acc_outcomes)
            res["system_a"]["explainability_ci"] = compute_bootstrap_ci(a_exp_outcomes)

            # Update System B stats
            if "seeds" in res["system_b"] and "seed_42" in res["system_b"]["seeds"]:
                b_run = res["system_b"]["seeds"]["seed_42"]
                b_items = b_run.get("items", [])
                if b_items:
                    b_acc_outcomes = [item["accuracy_correct"] for item in b_items]
                    b_exp_outcomes = [1 if item.get("explainability", {}).get("citation_correct") else 0 for item in b_items]
                    
                    b_run["accuracy_ci"] = compute_bootstrap_ci(b_acc_outcomes, random_seed=42)
                    b_run["explainability_ci"] = compute_bootstrap_ci(b_exp_outcomes, random_seed=42)
                    
                    mcn_acc = mcnemar_test(a_acc_outcomes, b_acc_outcomes)
                    mcn_exp = mcnemar_test(a_exp_outcomes, b_exp_outcomes)
                    
                    res["statistical_tests"] = {
                        "mcnemar_accuracy": mcn_acc,
                        "mcnemar_explainability": mcn_exp,
                        "effect_size_accuracy": compute_effect_size(mcn_acc),
                        "effect_size_explainability": compute_effect_size(mcn_exp)
                    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print("Updated main_eval_results.json")

def update_policy_edit():
    file_path = RESULTS_DIR / "policy_edit_results.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_a = []
    all_b = []
    for domain, d_res in data.get("domain_results", {}).items():
        dom_a = [item["system_a_correct"] for item in d_res.get("items", [])]
        dom_b = [item["system_b_correct"] for item in d_res.get("items", [])]
        all_a.extend(dom_a)
        all_b.extend(dom_b)
        d_res["system_a_accuracy"] = compute_bootstrap_ci(dom_a)
        d_res["system_b_accuracy"] = compute_bootstrap_ci(dom_b)
    
    if all_a and all_b:
        data["overall_system_a"] = compute_bootstrap_ci(all_a)
        data["overall_system_b"] = compute_bootstrap_ci(all_b)
        data["mcnemar_test"] = mcnemar_test(all_a, all_b)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print("Updated policy_edit_results.json")

if __name__ == "__main__":
    update_main_eval()
    update_policy_edit()
