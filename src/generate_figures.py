"""Generate Publication-Quality Figures and Tables for the GovRe Research Study.
Outputs charts to results/ and summary markdown tables.
"""

import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR

def generate_all_plots():
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})

    # Load results
    main_res_file = RESULTS_DIR / "main_eval_results.json"
    edit_res_file = RESULTS_DIR / "policy_edit_results.json"

    with open(main_res_file, "r", encoding="utf-8") as f:
        main_data = json.load(f)

    with open(edit_res_file, "r", encoding="utf-8") as f:
        edit_data = json.load(f)

    domains = ["hr", "finance", "it_access"]
    domain_labels = ["HR Operations", "Expense / Finance", "IT Access Security"]

    # 1. Main Experiment Accuracy Comparison Plot
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    x = np.arange(len(domains))
    width = 0.35

    a_accs = [main_data[d]["system_a"]["accuracy_ci"]["mean"] * 100 for d in domains]
    a_errs = [
        [
            (main_data[d]["system_a"]["accuracy_ci"]["mean"] - main_data[d]["system_a"]["accuracy_ci"]["ci_lower"]) * 100,
            (main_data[d]["system_a"]["accuracy_ci"]["ci_upper"] - main_data[d]["system_a"]["accuracy_ci"]["mean"]) * 100,
        ]
        for d in domains
    ]
    a_errs = np.array(a_errs).T

    b_accs = [main_data[d]["system_b"]["mean_accuracy"] * 100 for d in domains]
    b_errs = [
        [
            (main_data[d]["system_b"]["seeds"]["seed_42"]["accuracy_ci"]["mean"] - main_data[d]["system_b"]["seeds"]["seed_42"]["accuracy_ci"]["ci_lower"]) * 100,
            (main_data[d]["system_b"]["seeds"]["seed_42"]["accuracy_ci"]["ci_upper"] - main_data[d]["system_b"]["seeds"]["seed_42"]["accuracy_ci"]["mean"]) * 100,
        ]
        for d in domains
    ]
    b_errs = np.array(b_errs).T

    rects1 = ax.bar(x - width/2, a_accs, width, yerr=a_errs, capsize=5, label='System A (Rule-Based Baseline)', color='#2b5c8f', alpha=0.9)
    rects2 = ax.bar(x + width/2, b_accs, width, yerr=b_errs, capsize=5, label='System B (Retrieval-Grounded)', color='#e06666', alpha=0.9)

    ax.set_ylabel('Decision Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Governance Decision Accuracy Across Three Domains (with 95% Bootstrap CIs)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(domain_labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig_path = RESULTS_DIR / "accuracy_comparison.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path}")

    # 2. Policy Edit Robustness Experiment Plot
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    labels = ['Pre-Edit Baseline', 'Post-Policy-Edit (Stale Code vs Re-index)']
    x_edit = np.arange(len(labels))

    # Pre-edit average accuracy
    pre_a = np.mean(a_accs)
    pre_b = np.mean(b_accs)

    # Post-edit accuracy
    post_a = edit_data["overall_system_a"]["mean"] * 100
    post_b = edit_data["overall_system_b"]["mean"] * 100

    a_vals = [pre_a, post_a]
    b_vals = [pre_b, post_b]

    rects1 = ax.bar(x_edit - width/2, a_vals, width, label='System A (Rule-Based Code)', color='#2b5c8f', alpha=0.9)
    rects2 = ax.bar(x_edit + width/2, b_vals, width, label='System B (Retrieval-Grounded)', color='#27ae60', alpha=0.9)

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Policy-Change Robustness: Zero-Code Dynamic Adaptation vs Stale Code Failure', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x_edit)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig_edit_path = RESULTS_DIR / "policy_edit_robustness.png"
    plt.savefig(fig_edit_path)
    plt.close()
    print(f"Saved: {fig_edit_path}")

    # 3. Export Summary Table (CSV & MD)
    table_rows = []
    for d, d_name in zip(domains, domain_labels):
        d_res = main_data[d]
        a_acc = d_res["system_a"]["accuracy_ci"]
        a_exp = d_res["system_a"]["explainability_ci"]
        b_acc = d_res["system_b"]["seeds"]["seed_42"]["accuracy_ci"]
        b_exp = d_res["system_b"]["seeds"]["seed_42"]["explainability_ci"]
        mc_acc = d_res["statistical_tests"]["mcnemar_accuracy"]

        table_rows.append({
            "Domain": d_name,
            "N": d_res["n_scenarios"],
            "System A Acc (%)": f"{a_acc['mean']*100:.1f}% [{a_acc['ci_lower']*100:.1f}%, {a_acc['ci_upper']*100:.1f}%]",
            "System B Acc (%)": f"{b_acc['mean']*100:.1f}% [{b_acc['ci_lower']*100:.1f}%, {b_acc['ci_upper']*100:.1f}%]",
            "McNemar p-val": mc_acc["p_value"],
            "Significant (p<0.05)": mc_acc["is_statistically_significant"],
            "System A Explain (%)": f"{a_exp['mean']*100:.1f}%",
            "System B Explain (%)": f"{b_exp['mean']*100:.1f}%",
        })

    df = pd.DataFrame(table_rows)
    df.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    with open(RESULTS_DIR / "summary_table.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    print("Exported summary table to CSV and Markdown.")

if __name__ == "__main__":
    generate_all_plots()
