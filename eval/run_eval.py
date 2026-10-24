"""Main Evaluation Runner for Governance Grounding Research.

Evaluates System A (Rule-Based) and System B (Retrieval-Grounded) across
HR, Finance, and IT Access domains with:
- Bootstrap 95% Confidence Intervals (BCa method)
- McNemar's Paired Significance Tests
- Effect Size reporting (Odds Ratio, Cohen's g)
- Response source tracking (filters out API failures)
- Retrieval quality analysis (Recall@k, retrieval vs reasoning failures)
- Per-system latency profiling (mean, P50, P95, P99)
- Stratified accuracy by decision class (APPROVE/FLAG/BLOCK)
- Confusion matrices per system per domain
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter
import pandas as pd
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DOMAINS, SCENARIOS_FILES, RESULTS_DIR, EVAL_SEEDS
from src.rule_engine import evaluate_hr_rule, evaluate_finance_rule, evaluate_it_access_rule
from src.retrieval_governance import RetrievalGovernanceJudge
from src.hybrid_router import HybridGovernanceRouter
from src.explainability_scorer import score_system_a_explainability, score_system_b_explainability
from stats.significance_tests import compute_bootstrap_ci, mcnemar_test

RULE_EVALUATORS = {
    "hr": evaluate_hr_rule,
    "finance": evaluate_finance_rule,
    "it_access": evaluate_it_access_rule,
}

DECISION_LABELS = ["APPROVE", "FLAG", "BLOCK"]


def load_scenarios(domain: str) -> List[Dict[str, Any]]:
    path = SCENARIOS_FILES[domain]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_confusion_matrix(
    gold_labels: List[str], pred_labels: List[str]
) -> Dict[str, Dict[str, int]]:
    """Computes 3x3 confusion matrix for APPROVE/FLAG/BLOCK."""
    matrix: Dict[str, Dict[str, int]] = {}
    for gold in DECISION_LABELS:
        matrix[gold] = {}
        for pred in DECISION_LABELS:
            matrix[gold][pred] = sum(
                1 for g, p in zip(gold_labels, pred_labels)
                if g == gold and p == pred
            )
    return matrix


def compute_stratified_accuracy(
    gold_labels: List[str], pred_labels: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Computes accuracy stratified by gold decision class."""
    stratified = {}
    for cls in DECISION_LABELS:
        indices = [i for i, g in enumerate(gold_labels) if g == cls]
        if not indices:
            stratified[cls] = {"n": 0, "accuracy": 0.0, "correct": 0}
            continue
        correct = sum(1 for i in indices if gold_labels[i] == pred_labels[i])
        stratified[cls] = {
            "n": len(indices),
            "correct": correct,
            "accuracy": round(correct / len(indices), 4),
        }
    return stratified


def compute_retrieval_recall(
    scenarios: List[Dict[str, Any]], sys_b_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Computes Retrieval Recall@k: fraction of scenarios where gold_chunk_id
    appears in retrieved_chunks."""
    total = 0
    hit_at_1 = 0
    hit_at_k = 0
    gold_missing = 0

    for sc, item in zip(scenarios, sys_b_items):
        gold_chunk = sc.get("gold_chunk_id", "")
        if not gold_chunk:
            gold_missing += 1
            continue
        total += 1
        retrieved = item.get("retrieved_chunks", [])
        if retrieved and gold_chunk in retrieved[0]:
            hit_at_1 += 1
        if any(gold_chunk in r for r in retrieved):
            hit_at_k += 1

    return {
        "total_with_gold": total,
        "gold_missing": gold_missing,
        "recall_at_1": round(hit_at_1 / total, 4) if total > 0 else 0.0,
        "recall_at_k": round(hit_at_k / total, 4) if total > 0 else 0.0,
        "hits_at_1": hit_at_1,
        "hits_at_k": hit_at_k,
    }


def compute_latency_profile(latencies_ms: List[float]) -> Dict[str, float]:
    """Computes latency statistics: mean, P50, P95, P99."""
    if not latencies_ms:
        return {"n_samples": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    arr = np.array(latencies_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "n_samples": len(latencies_ms),
    }


def compute_effect_size(mcnemar_result: Dict[str, Any]) -> Dict[str, Any]:
    """Computes effect sizes for McNemar's test: Odds Ratio and Cohen's g."""
    b = mcnemar_result["b_discordant_a_wins"]
    c = mcnemar_result["c_discordant_b_wins"]
    discordant = b + c

    if c > 0:
        odds_ratio = round(b / c, 4)
    elif b > 0:
        odds_ratio = float("inf")
    else:
        odds_ratio = 1.0

    if discordant > 0:
        cohens_g = round(max(b, c) / discordant - 0.5, 4)
    else:
        cohens_g = 0.0

    return {
        "odds_ratio": odds_ratio,
        "cohens_g": cohens_g,
        "b_discordant_a_wins": b,
        "c_discordant_b_wins": c,
    }


def evaluate_domain(domain: str, seeds: List[int] = EVAL_SEEDS) -> Dict[str, Any]:
    scenarios = load_scenarios(domain)
    rule_fn = RULE_EVALUATORS[domain]
    judge = RetrievalGovernanceJudge(domain=domain, is_edited=False)

    print(f"\n=======================================================")
    print(f" EVALUATING DOMAIN: {domain.upper()} ({len(scenarios)} Scenarios)")
    print(f"=======================================================")

    # ================================================================
    # 1. Evaluate System A (Rule-Based) with latency timing
    # ================================================================
    sys_a_results = []
    sys_a_acc_outcomes = []
    sys_a_exp_outcomes = []
    sys_a_gold_labels = []
    sys_a_pred_labels = []
    sys_a_latencies = []

    for sc in scenarios:
        t0 = time.perf_counter()
        pred_a = rule_fn(sc)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000

        gold_dec = sc["gold_decision"].upper().strip()
        pred_dec = pred_a["decision"].upper().strip()
        is_acc_correct = int(pred_dec == gold_dec)
        exp_score = score_system_a_explainability(pred_a, sc)
        is_exp_correct = int(exp_score["citation_correct"])

        sys_a_acc_outcomes.append(is_acc_correct)
        sys_a_exp_outcomes.append(is_exp_correct)
        sys_a_gold_labels.append(gold_dec)
        sys_a_pred_labels.append(pred_dec)
        sys_a_latencies.append(latency_ms)

        sys_a_results.append({
            "scenario_id": sc["id"],
            "gold_decision": gold_dec,
            "pred_decision": pred_dec,
            "accuracy_correct": is_acc_correct,
            "rule_id": pred_a.get("rule_id", ""),
            "reason": pred_a.get("reason", ""),
            "explainability": exp_score,
            "latency_ms": round(latency_ms, 4),
        })

    sys_a_acc_ci = compute_bootstrap_ci(sys_a_acc_outcomes)
    sys_a_exp_ci = compute_bootstrap_ci(sys_a_exp_outcomes)
    sys_a_confusion = compute_confusion_matrix(sys_a_gold_labels, sys_a_pred_labels)
    sys_a_stratified = compute_stratified_accuracy(sys_a_gold_labels, sys_a_pred_labels)
    sys_a_latency_profile = compute_latency_profile(sys_a_latencies)

    # ================================================================
    # 2. Evaluate System B (Retrieval-Grounded) across seeds
    # ================================================================
    seed_runs = {}
    aggregated_b_acc = []
    aggregated_b_exp = []

    for seed in seeds:
        print(f"\n--- Running System B with Seed {seed} ---")
        run_acc_outcomes = []
        run_exp_outcomes = []
        run_items = []
        run_gold_labels = []
        run_pred_labels = []
        run_latencies = []
        run_sources = Counter()
        run_api_failures = 0

        for idx, sc in enumerate(scenarios, 1):
            pred_b = judge.evaluate_scenario(scenario=sc, temperature=0.0, seed=seed)
            gold_dec = sc["gold_decision"].upper().strip()
            pred_dec = pred_b["decision"].upper().strip()
            response_source = pred_b.get("response_source", "unknown")

            run_sources[response_source] += 1

            if response_source == "api_failure":
                run_api_failures += 1

            is_acc_correct = int(pred_dec == gold_dec)
            exp_score = score_system_b_explainability(pred_b, sc)
            is_exp_correct = int(exp_score["citation_correct"])

            run_acc_outcomes.append(is_acc_correct)
            run_exp_outcomes.append(is_exp_correct)
            run_gold_labels.append(gold_dec)
            run_pred_labels.append(pred_dec)
            run_latencies.append(pred_b.get("latency_ms", 0.0))

            run_items.append({
                "scenario_id": sc["id"],
                "gold_decision": gold_dec,
                "pred_decision": pred_dec,
                "accuracy_correct": is_acc_correct,
                "citation_chunk_id": pred_b.get("citation_chunk_id", ""),
                "citation_text": pred_b.get("citation_text", ""),
                "reasoning": pred_b.get("reasoning", ""),
                "explainability": exp_score,
                "response_source": response_source,
                "model_used": pred_b.get("model_used", ""),
                "latency_ms": pred_b.get("latency_ms", 0.0),
                "prompt_tokens": pred_b.get("prompt_tokens", 0),
                "completion_tokens": pred_b.get("completion_tokens", 0),
                "total_tokens": pred_b.get("total_tokens", 0),
                "retrieved_chunks": pred_b.get("retrieved_chunks", []),
                "api_errors": pred_b.get("api_errors", []),
            })

            source_tag = f" [{response_source}]" if response_source != "llm" else ""
            print(
                f"  [{idx:02d}/{len(scenarios)}] {sc['id']}: Gold={gold_dec} | "
                f"Sys B={pred_dec} ({'CORRECT' if is_acc_correct else 'INCORRECT'}) | "
                f"Citation={'OK' if is_exp_correct else 'MISS'}{source_tag}"
            )
            time.sleep(0.35)  # Rate limiting spacing

        # --- Evaluate System C (Hybrid) ---
        run_c_acc_outcomes = []
        run_c_latencies = []
        run_c_sources = Counter()
        router = HybridGovernanceRouter(domain=domain, judge=judge)
        for sc in scenarios:
            pred_c = router.evaluate_scenario(scenario=sc, temperature=0.0, seed=seed)
            gold_dec = sc["gold_decision"].upper().strip()
            pred_dec = pred_c["decision"].upper().strip()
            source = pred_c.get("response_source", "unknown")
            run_c_sources[source] += 1
            run_c_acc_outcomes.append(int(pred_dec == gold_dec))
            run_c_latencies.append(pred_c.get("latency_ms", 0.0))
            
        sys_c_acc = compute_bootstrap_ci(run_c_acc_outcomes, random_seed=seed)
        sys_c_lat = compute_latency_profile([l for l in run_c_latencies if l > 0])
        sys_c_src = dict(run_c_sources)


        # Compute retrieval recall for this seed run
        retrieval_recall = compute_retrieval_recall(scenarios, run_items)

        # Separate LLM-only outcomes for fair comparison
        llm_only_acc = [
            run_acc_outcomes[i] for i, item in enumerate(run_items)
            if item["response_source"] == "llm"
        ]
        llm_only_exp = [
            run_exp_outcomes[i] for i, item in enumerate(run_items)
            if item["response_source"] == "llm"
        ]
        llm_only_latencies = [
            item["latency_ms"] for item in run_items
            if item["response_source"] == "llm" and item["latency_ms"] > 0
        ]

        seed_runs[f"seed_{seed}"] = {
            "items": run_items,
            "system_c_accuracy_ci": sys_c_acc,
            "system_c_latency_profile": sys_c_lat,
            "system_c_sources": sys_c_src,
            "accuracy_ci": compute_bootstrap_ci(run_acc_outcomes, random_seed=seed),
            "accuracy_ci_llm_only": compute_bootstrap_ci(llm_only_acc, random_seed=seed) if llm_only_acc else None,
            "explainability_ci": compute_bootstrap_ci(run_exp_outcomes, random_seed=seed),
            "explainability_ci_llm_only": compute_bootstrap_ci(llm_only_exp, random_seed=seed) if llm_only_exp else None,
            "response_sources": dict(run_sources),
            "api_failure_count": run_api_failures,
            "api_failure_rate": round(run_api_failures / len(scenarios), 4) if scenarios else 0.0,
            "retrieval_recall": retrieval_recall,
            "confusion_matrix": compute_confusion_matrix(run_gold_labels, run_pred_labels),
            "stratified_accuracy": compute_stratified_accuracy(run_gold_labels, run_pred_labels),
            "latency_profile": compute_latency_profile(llm_only_latencies),
        }
        aggregated_b_acc.append(run_acc_outcomes)
        aggregated_b_exp.append(run_exp_outcomes)

    # Primary seed for paired testing
    primary_b_acc = aggregated_b_acc[0]
    primary_b_exp = aggregated_b_exp[0]
    primary_seed_key = f"seed_{seeds[0]}"

    # Paired McNemar Test + Effect Size
    mcnemar_acc = mcnemar_test(sys_a_acc_outcomes, primary_b_acc)
    mcnemar_exp = mcnemar_test(sys_a_exp_outcomes, primary_b_exp)
    effect_acc = compute_effect_size(mcnemar_acc)
    effect_exp = compute_effect_size(mcnemar_exp)

    # Summary printout
    primary_run = seed_runs[primary_seed_key]
    api_fail_rate = primary_run["api_failure_rate"]

    print(f"\n--- DOMAIN SUMMARY: {domain.upper()} ---")
    print(f"System A (Rule-Based):")
    print(f"  Decision Accuracy:       {sys_a_acc_ci['mean']*100:.1f}% [95% CI: {sys_a_acc_ci['ci_lower']*100:.1f}% - {sys_a_acc_ci['ci_upper']*100:.1f}%]")
    print(f"  Explainability Correct:  {sys_a_exp_ci['mean']*100:.1f}%")
    print(f"  Latency (mean):          {sys_a_latency_profile['mean_ms']:.3f} ms")

    b_mean_acc = np.mean([s["accuracy_ci"]["mean"] for s in seed_runs.values()])
    b_mean_exp = np.mean([s["explainability_ci"]["mean"] for s in seed_runs.values()])
    print(f"System B (Retrieval-Grounded):")
    print(f"  Decision Accuracy:       {b_mean_acc*100:.1f}% [Seed {seeds[0]} CI: {primary_run['accuracy_ci']['ci_lower']*100:.1f}% - {primary_run['accuracy_ci']['ci_upper']*100:.1f}%]")
    print(f"  Explainability Correct:  {b_mean_exp*100:.1f}%")
    if primary_run["accuracy_ci_llm_only"]:
        llm_acc = primary_run["accuracy_ci_llm_only"]["mean"]
        print(f"  LLM-Only Accuracy:       {llm_acc*100:.1f}% (excluding {primary_run['api_failure_count']} API failures)")
    print(f"  API Failure Rate:        {api_fail_rate*100:.1f}%")
    print(f"  Retrieval Recall@k:      {primary_run['retrieval_recall']['recall_at_k']*100:.1f}%")
    if primary_run["latency_profile"]["n_samples"] > 0:
        print(f"  Latency (P50/P95):       {primary_run['latency_profile']['p50_ms']:.0f}ms / {primary_run['latency_profile']['p95_ms']:.0f}ms")

    print(f"System C (Hybrid Router):")
    print(f"  Decision Accuracy:       {np.mean([s['system_c_accuracy_ci']['mean'] for s in seed_runs.values()])*100:.1f}% [Seed {seeds[0]} CI: {primary_run['system_c_accuracy_ci']['ci_lower']*100:.1f}% - {primary_run['system_c_accuracy_ci']['ci_upper']*100:.1f}%]")
    if primary_run["system_c_latency_profile"]["n_samples"] > 0:
        print(f"  Latency (P50/P95):       {primary_run['system_c_latency_profile']['p50_ms']:.0f}ms / {primary_run['system_c_latency_profile']['p95_ms']:.0f}ms")


    print(f"McNemar Test (Accuracy):   p-value = {mcnemar_acc['p_value']} (Significant: {mcnemar_acc['is_statistically_significant']})")
    print(f"Effect Size (Odds Ratio):  {effect_acc['odds_ratio']}  |  Cohen's g: {effect_acc['cohens_g']}")

    if api_fail_rate > 0.1:
        print(f"\n  *** WARNING: {api_fail_rate*100:.0f}% of System B responses came from API failures. ***")
        print(f"  *** Results may not reflect true retrieval-grounded governance performance. ***")

    return {
        "domain": domain,
        "n_scenarios": len(scenarios),
        "system_a": {
            "accuracy_ci": sys_a_acc_ci,
            "explainability_ci": sys_a_exp_ci,
            "confusion_matrix": sys_a_confusion,
            "stratified_accuracy": sys_a_stratified,
            "latency_profile": sys_a_latency_profile,
            "items": sys_a_results,
        },
        "system_b": {
            "seeds": seed_runs,
            "mean_accuracy": float(b_mean_acc),
            "mean_explainability": float(b_mean_exp),
        },
        "system_c": {
            "mean_accuracy": float(np.mean([s["system_c_accuracy_ci"]["mean"] for s in seed_runs.values()])),
        },
        "statistical_tests": {
            "mcnemar_accuracy": mcnemar_acc,
            "mcnemar_explainability": mcnemar_exp,
            "effect_size_accuracy": effect_acc,
            "effect_size_explainability": effect_exp,
        }
    }


def run_all_evaluations() -> Dict[str, Any]:
    all_results = {}
    for domain in DOMAINS:
        all_results[domain] = evaluate_domain(domain)

    # Generate summary table
    _generate_summary_table(all_results)

    # Save full results json
    out_file = RESULTS_DIR / "main_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll domain evaluations complete. Results saved to: {out_file}")
    return all_results


def _generate_summary_table(all_results: Dict[str, Any]) -> None:
    """Generates summary_table.md and summary_table.csv from evaluation results."""
    domain_display = {
        "hr": "HR Operations",
        "finance": "Expense / Finance",
        "it_access": "IT Access Security",
    }

    rows = []
    for domain in DOMAINS:
        res = all_results[domain]
        n = res["n_scenarios"]
        a_ci = res["system_a"]["accuracy_ci"]
        primary_seed_key = f"seed_{EVAL_SEEDS[0]}"
        b_run = res["system_b"]["seeds"][primary_seed_key]
        b_ci = b_run["accuracy_ci"]
        mcn = res["statistical_tests"]["mcnemar_accuracy"]
        eff = res["statistical_tests"]["effect_size_accuracy"]

        a_exp = res["system_a"]["explainability_ci"]["mean"]
        b_exp = b_run["explainability_ci"]["mean"]
        api_fail = b_run["api_failure_rate"]
        recall_k = b_run["retrieval_recall"]["recall_at_k"]

        rows.append({
            "Domain": domain_display.get(domain, domain),
            "N": n,
            "System A Acc (%)": f"{a_ci['mean']*100:.1f}% [{a_ci['ci_lower']*100:.1f}%, {a_ci['ci_upper']*100:.1f}%]",
            "System B Acc (%)": f"{b_ci['mean']*100:.1f}% [{b_ci['ci_lower']*100:.1f}%, {b_ci['ci_upper']*100:.1f}%]",
            "System C Acc (%)": f"{b_run['system_c_accuracy_ci']['mean']*100:.1f}% [{b_run['system_c_accuracy_ci']['ci_lower']*100:.1f}%, {b_run['system_c_accuracy_ci']['ci_upper']*100:.1f}%]",
            "McNemar p-val": f"{mcn['p_value']:.5f}",
            "Significant (p<0.05)": str(mcn["is_statistically_significant"]),
            "Odds Ratio": f"{eff['odds_ratio']}",
            "System A Explain (%)": f"{a_exp*100:.1f}%",
            "System B Explain (%)": f"{b_exp*100:.1f}%",
            "API Fail Rate": f"{api_fail*100:.1f}%",
            "Retrieval Recall@k": f"{recall_k*100:.1f}%",
        })

    df = pd.DataFrame(rows)
    md_table = df.to_markdown(index=False)
    csv_path = RESULTS_DIR / "summary_table.csv"
    md_path = RESULTS_DIR / "summary_table.md"

    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table)

    print(f"\nSummary table saved to: {md_path} and {csv_path}")


if __name__ == "__main__":
    run_all_evaluations()
